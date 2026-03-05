#!/usr/bin/env bun

import http from "node:http";
import net from "node:net";
import { Readable } from "node:stream";
import { spawn } from "node:child_process";
import { accessSync, constants, existsSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import {
  applyDefaultEffort,
  approxTokenCount,
  hasEffortFlag,
  parseChatgptRefreshConfigFromAuthJson,
  parseChatgptTokenFromAuthJson,
  parseClaudexArgs,
  parseApiKeyFromAuthJson,
  parseCodexConfig,
  resolveUpstreamFromCodexConfig,
  sanitizeToolFields,
  type JsonObject,
} from "./core.ts";

const rawArgs = process.argv.slice(2);
const parsedArgs = parseClaudexArgs(rawArgs);
const args = parsedArgs.claudeArgs;
const hasSettingsArg = parsedArgs.hasSettingsArg;
const safeMode = parsedArgs.safeMode;
const preserveClientEffort = hasEffortFlag(args);
const defaultReasoningEffort = process.env.CLAUDEX_DEFAULT_REASONING_EFFORT || "xhigh";
const debug = process.env.CLAUDEX_DEBUG === "1";
const claudeSubcommands = new Set([
  "agents",
  "auth",
  "doctor",
  "install",
  "mcp",
  "open",
  "plugin",
  "server",
  "setup-token",
  "update",
  "upgrade",
  "remote-control",
  "rc",
]);

interface RuntimeConfig {
  upstreamBaseUrl: string;
  upstreamBearerToken: string;
  upstreamExtraHeaders: Record<string, string>;
  forcedModel: string;
  authMode: "provider-api-key" | "chatgpt-token" | "chatgpt-api-key";
  chatgptRefreshConfig?: {
    authPath: string;
    refreshToken: string;
    clientId: string;
  };
}

interface ProxyOptions {
  forcedModel: string;
  defaultReasoningEffort: string;
  preserveClientEffort: boolean;
  debug: boolean;
}

interface UpstreamAuthState {
  bearerToken: string;
  extraHeaders: Record<string, string>;
  chatgptRefreshConfig?: {
    authPath: string;
    refreshToken: string;
    clientId: string;
  };
  refreshInFlight?: Promise<string>;
}

function fail(message: string): never {
  console.error(`claudex: ${message}`);
  process.exit(1);
}

function ensureExecutable(path: string): boolean {
  try {
    accessSync(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function safeRealpath(path: string): string | null {
  try {
    return realpathSync(path);
  } catch {
    return null;
  }
}

function resolveSelfDir(): string {
  const arg1 = process.argv[1];
  if (typeof arg1 === "string" && arg1.length > 0 && !arg1.startsWith("-")) {
    const resolvedArg1 = safeRealpath(arg1);
    if (resolvedArg1) {
      return dirname(resolvedArg1);
    }
  }

  const resolvedExecPath = safeRealpath(process.execPath);
  if (resolvedExecPath) {
    return dirname(resolvedExecPath);
  }

  return process.cwd();
}

function resolveCodexPaths(): { configPath: string; authPath: string } {
  const codexHome = process.env.CODEX_HOME?.trim() || join(homedir(), ".codex");
  const configPath = process.env.CLAUDEX_CODEX_CONFIG?.trim() || join(codexHome, "config.toml");
  const authPath = process.env.CLAUDEX_CODEX_AUTH?.trim() || join(codexHome, "auth.json");
  return { configPath, authPath };
}

function loadRuntimeConfig(): RuntimeConfig {
  const { configPath, authPath } = resolveCodexPaths();

  const providerOverride = process.env.CLAUDEX_MODEL_PROVIDER;
  const baseUrlOverride = process.env.CLAUDEX_UPSTREAM_BASE_URL;
  const chatgptBaseUrl =
    process.env.CLAUDEX_CHATGPT_BASE_URL?.trim() || "https://chatgpt.com/backend-api/codex";
  const envApiKey = process.env.CLAUDEX_UPSTREAM_API_KEY || process.env.OPENAI_API_KEY;
  const envBearerToken =
    process.env.CLAUDEX_UPSTREAM_BEARER_TOKEN || process.env.CLAUDEX_CHATGPT_BEARER_TOKEN;
  const envChatgptAccountId = process.env.CLAUDEX_CHATGPT_ACCOUNT_ID;

  let configContents = "";
  let modelFromConfig: string | undefined;
  let resolvedProvider: ReturnType<typeof resolveUpstreamFromCodexConfig> | null = null;
  if (existsSync(configPath)) {
    configContents = readFileSync(configPath, "utf8");
    modelFromConfig = parseCodexConfig(configContents).model;
    try {
      resolvedProvider = resolveUpstreamFromCodexConfig(configContents, {
        providerOverride,
        baseUrlOverride,
      });
    } catch {
      resolvedProvider = null;
    }
  } else if (baseUrlOverride?.trim()) {
    resolvedProvider = {
      baseUrl: baseUrlOverride.trim(),
      providerKey: providerOverride,
      model: undefined,
    };
  }

  const forcedModel = (process.env.CLAUDEX_FORCE_MODEL || modelFromConfig || "gpt-5.3-codex").trim();

  const authFileExists = existsSync(authPath);
  const authContents = authFileExists ? readFileSync(authPath, "utf8") : "";
  const authUnavailable = !authFileExists && !envApiKey?.trim() && !envBearerToken?.trim();
  if (authUnavailable) {
    fail(`missing auth file: ${authPath}`);
  }

  if (resolvedProvider?.baseUrl?.trim()) {
    const upstreamBearerToken = parseApiKeyFromAuthJson(authContents, envApiKey);
    return {
      upstreamBaseUrl: resolvedProvider.baseUrl,
      upstreamBearerToken,
      upstreamExtraHeaders: {},
      forcedModel,
      authMode: "provider-api-key",
    };
  }

  try {
    const tokenAuth = parseChatgptTokenFromAuthJson(authContents, {
      envBearerToken,
      envAccountId: envChatgptAccountId,
    });
    const refreshConfig = authContents.trim().length > 0 ? parseChatgptRefreshConfigFromAuthJson(authContents) : {};

    const extraHeaders: Record<string, string> = {};
    if (tokenAuth.accountId) {
      extraHeaders["chatgpt-account-id"] = tokenAuth.accountId;
    }

    const canAutoRefresh =
      !envBearerToken?.trim() &&
      typeof refreshConfig.refreshToken === "string" &&
      refreshConfig.refreshToken.length > 0 &&
      typeof refreshConfig.clientId === "string" &&
      refreshConfig.clientId.length > 0;

    return {
      upstreamBaseUrl: chatgptBaseUrl,
      upstreamBearerToken: tokenAuth.bearerToken,
      upstreamExtraHeaders: extraHeaders,
      forcedModel,
      authMode: "chatgpt-token",
      chatgptRefreshConfig: canAutoRefresh
        ? {
            authPath,
            refreshToken: refreshConfig.refreshToken!,
            clientId: refreshConfig.clientId!,
          }
        : undefined,
    };
  } catch {
    const upstreamBearerToken = parseApiKeyFromAuthJson(authContents, envApiKey);
    return {
      upstreamBaseUrl: chatgptBaseUrl,
      upstreamBearerToken,
      upstreamExtraHeaders: {},
      forcedModel,
      authMode: "chatgpt-api-key",
    };
  }
}

function resolveClaudeBinary(): string {
  if (process.env.CLAUDEX_CLAUDE_BIN) {
    return process.env.CLAUDEX_CLAUDE_BIN;
  }

  const scriptDir = resolveSelfDir();
  const reverseDir = join(scriptDir, "reverse");

  if (existsSync(reverseDir)) {
    const localCandidates = readdirSync(reverseDir, { withFileTypes: true })
      .filter((entry) => entry.isFile() && entry.name.startsWith("claude-"))
      .map((entry) => join(reverseDir, entry.name))
      .filter((path) => ensureExecutable(path))
      .sort((a, b) => b.localeCompare(a, undefined, { numeric: true, sensitivity: "base" }));

    if (localCandidates.length > 0) {
      return localCandidates[0];
    }
  }

  const binaryInPath = Bun.which("claude");
  if (binaryInPath) {
    return binaryInPath;
  }

  fail("Claude binary not found. Set CLAUDEX_CLAUDE_BIN.");
}

function pickFreePort(host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, host, () => {
      const addr = server.address();
      if (!addr || typeof addr === "string") {
        server.close();
        reject(new Error("failed to get listen port"));
        return;
      }

      const port = addr.port;
      server.close((closeError) => {
        if (closeError) {
          reject(closeError);
          return;
        }
        resolve(port);
      });
    });
  });
}

function writeJson(res: http.ServerResponse, statusCode: number, data: unknown): void {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer | string) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function copyHeadersFromUpstream(headers: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  headers.forEach((value, key) => {
    if (key.toLowerCase() === "transfer-encoding") {
      return;
    }
    out[key] = value;
  });
  return out;
}

function normalizeBasePath(pathname: string): string {
  if (pathname === "/" || pathname.trim().length === 0) {
    return "";
  }
  return `/${pathname.replace(/^\/+/, "").replace(/\/+$/, "")}`;
}

function buildUpstreamUrl(upstreamOrigin: URL, requestPath: string): URL {
  const incoming = new URL(requestPath, "http://localhost");
  const basePath = normalizeBasePath(upstreamOrigin.pathname);

  let resolvedPath = incoming.pathname;
  if (basePath && resolvedPath !== basePath && !resolvedPath.startsWith(`${basePath}/`)) {
    resolvedPath = `${basePath}${resolvedPath.startsWith("/") ? "" : "/"}${resolvedPath}`;
  }

  const upstream = new URL(upstreamOrigin.toString());
  upstream.pathname = resolvedPath;
  upstream.search = incoming.search;
  return upstream;
}

function persistRefreshedChatgptTokens(
  authPath: string,
  refreshed: {
    accessToken: string;
    idToken?: string;
    refreshToken?: string;
  },
  options: {
    debug: boolean;
  }
): void {
  try {
    const current = existsSync(authPath) ? readFileSync(authPath, "utf8") : "{}";
    const parsed = JSON.parse(current) as Record<string, any>;
    if (!parsed.tokens || typeof parsed.tokens !== "object") {
      parsed.tokens = {};
    }

    parsed.tokens.access_token = refreshed.accessToken;
    if (typeof refreshed.idToken === "string" && refreshed.idToken.length > 0) {
      parsed.tokens.id_token = refreshed.idToken;
    }
    if (typeof refreshed.refreshToken === "string" && refreshed.refreshToken.length > 0) {
      parsed.tokens.refresh_token = refreshed.refreshToken;
    }
    parsed.last_refresh = new Date().toISOString();

    writeFileSync(authPath, `${JSON.stringify(parsed, null, 2)}\n`, "utf8");
  } catch (error) {
    if (options.debug) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`claudex: warning: failed to persist refreshed auth tokens: ${message}`);
    }
  }
}

async function refreshChatgptBearerToken(
  authState: UpstreamAuthState,
  options: {
    debug: boolean;
  }
): Promise<string> {
  if (!authState.chatgptRefreshConfig) {
    throw new Error("chatgpt refresh config is missing");
  }
  if (authState.refreshInFlight) {
    return authState.refreshInFlight;
  }

  authState.refreshInFlight = (async () => {
    const refreshConfig = authState.chatgptRefreshConfig!;
    const response = await fetch("https://auth.openai.com/oauth/token", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        grant_type: "refresh_token",
        refresh_token: refreshConfig.refreshToken,
        client_id: refreshConfig.clientId,
      }),
    });

    const text = await response.text();
    if (!response.ok) {
      throw new Error(`token refresh failed with status ${response.status}: ${text.slice(0, 200)}`);
    }

    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error("token refresh returned non-JSON response");
    }

    const refreshedAccessToken =
      typeof parsed.access_token === "string" && parsed.access_token.trim().length > 0
        ? parsed.access_token.trim()
        : undefined;
    const refreshedIdToken =
      typeof parsed.id_token === "string" && parsed.id_token.trim().length > 0
        ? parsed.id_token.trim()
        : undefined;
    const refreshedRefreshToken =
      typeof parsed.refresh_token === "string" && parsed.refresh_token.trim().length > 0
        ? parsed.refresh_token.trim()
        : undefined;

    const nextBearerToken = refreshedAccessToken || refreshedIdToken;
    if (!nextBearerToken) {
      throw new Error("token refresh response did not include access_token or id_token");
    }

    authState.bearerToken = nextBearerToken;
    if (refreshedRefreshToken) {
      authState.chatgptRefreshConfig = {
        ...authState.chatgptRefreshConfig!,
        refreshToken: refreshedRefreshToken,
      };
    }

    persistRefreshedChatgptTokens(
      refreshConfig.authPath,
      {
        accessToken: refreshedAccessToken || authState.bearerToken,
        idToken: refreshedIdToken,
        refreshToken: refreshedRefreshToken,
      },
      options
    );

    if (options.debug) {
      console.error("claudex: refreshed ChatGPT bearer token using refresh_token");
    }
    return authState.bearerToken;
  })().finally(() => {
    authState.refreshInFlight = undefined;
  });

  return authState.refreshInFlight;
}

async function proxyRaw(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  bodyBuffer: Buffer,
  requestPath: string,
  upstreamOrigin: URL,
  authState: UpstreamAuthState,
  debugLogging: boolean,
  overrideBody: JsonObject | null = null,
  allowRefreshRetry = true
): Promise<void> {
  const headers: Record<string, string> = {
    authorization: `Bearer ${authState.bearerToken}`,
  };

  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) {
      continue;
    }
    const normalized = key.toLowerCase();
    if (normalized === "host" || normalized === "content-length" || normalized === "authorization") {
      continue;
    }
    headers[normalized] = Array.isArray(value) ? value.join(", ") : value;
  }
  for (const [key, value] of Object.entries(authState.extraHeaders)) {
    headers[key.toLowerCase()] = value;
  }

  let outboundBody = bodyBuffer;
  if (overrideBody !== null) {
    const payload = JSON.stringify(overrideBody);
    outboundBody = Buffer.from(payload);
    headers["content-type"] = "application/json";
    headers["content-length"] = String(outboundBody.length);
  } else if (outboundBody.length > 0) {
    headers["content-length"] = String(outboundBody.length);
  }

  const upstreamUrl = buildUpstreamUrl(upstreamOrigin, requestPath);
  const upstreamResponse = await fetch(upstreamUrl, {
    method: req.method,
    headers,
    body: req.method === "GET" || req.method === "HEAD" ? undefined : outboundBody,
  });

  if (upstreamResponse.status === 401 && allowRefreshRetry && authState.chatgptRefreshConfig) {
    try {
      await refreshChatgptBearerToken(authState, { debug: debugLogging });
      await proxyRaw(req, res, bodyBuffer, requestPath, upstreamOrigin, authState, debugLogging, overrideBody, false);
      return;
    } catch (error) {
      if (debugLogging) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`claudex: token refresh attempt failed: ${message}`);
      }
    }
  }

  res.writeHead(upstreamResponse.status, copyHeadersFromUpstream(upstreamResponse.headers));
  if (!upstreamResponse.body) {
    res.end();
    return;
  }

  Readable.fromWeb(upstreamResponse.body as any).pipe(res);
}

async function startProxy(
  listenHost: string,
  listenPort: number,
  upstreamOrigin: URL,
  upstreamBearerToken: string,
  upstreamExtraHeaders: Record<string, string>,
  chatgptRefreshConfig: RuntimeConfig["chatgptRefreshConfig"],
  options: ProxyOptions
): Promise<http.Server> {
  const authState: UpstreamAuthState = {
    bearerToken: upstreamBearerToken,
    extraHeaders: upstreamExtraHeaders,
    chatgptRefreshConfig,
  };

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url || "/", `http://${listenHost}:${listenPort}`);
      const path = url.pathname;
      const method = req.method || "GET";

      if (method === "GET" && path === "/health") {
        writeJson(res, 200, {
          ok: true,
          forced_model: options.forcedModel,
          upstream: upstreamOrigin.origin + upstreamOrigin.pathname,
        });
        return;
      }

      if (method === "GET" && path === "/v1/models") {
        writeJson(res, 200, {
          object: "list",
          data: [
            {
              id: options.forcedModel,
              object: "model",
              created: Math.floor(Date.now() / 1000),
              owned_by: "claudex",
            },
          ],
        });
        return;
      }

      if (method === "GET" && path.startsWith("/v1/models/")) {
        const requestedModel = decodeURIComponent(path.slice("/v1/models/".length));
        writeJson(res, 200, {
          id: requestedModel,
          object: "model",
          created: Math.floor(Date.now() / 1000),
          owned_by: "claudex",
        });
        return;
      }

      if (method === "POST" && path === "/v1/messages/count_tokens") {
        const bodyBuffer = await readBody(req);
        let parsed: JsonObject = {};
        try {
          parsed = bodyBuffer.length > 0 ? JSON.parse(bodyBuffer.toString("utf8")) : {};
        } catch {
          parsed = {};
        }
        writeJson(res, 200, { input_tokens: approxTokenCount(parsed) });
        return;
      }

      if (method === "POST" && path === "/v1/messages") {
        const bodyBuffer = await readBody(req);
        let parsed: JsonObject;
        try {
          parsed = bodyBuffer.length > 0 ? JSON.parse(bodyBuffer.toString("utf8")) : {};
        } catch {
          writeJson(res, 400, {
            type: "error",
            error: {
              type: "invalid_request_error",
              message: "Invalid JSON body",
            },
          });
          return;
        }

        const originalModel = String(parsed.model ?? "");
        parsed.model = options.forcedModel;
        applyDefaultEffort(parsed, {
          forcedModel: options.forcedModel,
          defaultReasoningEffort: options.defaultReasoningEffort,
          preserveClientEffort: options.preserveClientEffort,
        });
        const removedToolFields = sanitizeToolFields(parsed);

        if (options.debug) {
          const effort = parsed.output_config?.effort ?? parsed.reasoning?.effort ?? "unset";
          console.error(
            `claudex-proxy model remap: ${originalModel} -> ${options.forcedModel}, effort=${effort}, preserve_client_effort=${options.preserveClientEffort}, removed_tool_fields=${removedToolFields}`
          );
        }

        await proxyRaw(
          req,
          res,
          bodyBuffer,
          url.pathname + url.search,
          upstreamOrigin,
          authState,
          options.debug,
          parsed
        );
        return;
      }

      const bodyBuffer = await readBody(req);
      await proxyRaw(
        req,
        res,
        bodyBuffer,
        url.pathname + url.search,
        upstreamOrigin,
        authState,
        options.debug
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      writeJson(res, 500, {
        type: "error",
        error: {
          type: "api_error",
          message: `claudex-proxy internal error: ${message}`,
        },
      });
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listenPort, listenHost, () => resolve());
  });

  return server;
}

async function main(): Promise<void> {
  const runtime = loadRuntimeConfig();
  const upstreamOrigin = new URL(runtime.upstreamBaseUrl);
  const claudeBinary = resolveClaudeBinary();

  const listenHost = process.env.CLAUDEX_LISTEN_HOST || "127.0.0.1";
  const listenPort = process.env.CLAUDEX_PORT ? Number(process.env.CLAUDEX_PORT) : await pickFreePort(listenHost);
  if (!Number.isInteger(listenPort) || listenPort < 1 || listenPort > 65535) {
    fail("invalid CLAUDEX_PORT value");
  }

  const proxyServer = await startProxy(
    listenHost,
    listenPort,
    upstreamOrigin,
    runtime.upstreamBearerToken,
    runtime.upstreamExtraHeaders,
    runtime.chatgptRefreshConfig,
    {
      forcedModel: runtime.forcedModel,
      defaultReasoningEffort,
      preserveClientEffort,
      debug,
    }
  );

  const proxyUrl = `http://${listenHost}:${listenPort}`;
  const refreshStatus = runtime.chatgptRefreshConfig ? "on" : "off";
  console.error(
    `claudex: proxy=${proxyUrl} force_model=${runtime.forcedModel} safe_mode=${safeMode} auth_mode=${runtime.authMode} auto_refresh=${refreshStatus}`
  );

  const injectedArgs = [...args];
  const forcedLoginMethod = (process.env.CLAUDEX_FORCE_LOGIN_METHOD || "console").trim();
  const isSubcommandInvocation =
    injectedArgs.length > 0 && !injectedArgs[0].startsWith("-") && claudeSubcommands.has(injectedArgs[0]);
  if (!isSubcommandInvocation && !hasSettingsArg && forcedLoginMethod.length > 0 && forcedLoginMethod !== "none") {
    injectedArgs.push("--settings", JSON.stringify({ forceLoginMethod: forcedLoginMethod }));
  }

  const childEnv: NodeJS.ProcessEnv = {
    ...process.env,
    ANTHROPIC_BASE_URL: proxyUrl,
    ANTHROPIC_API_KEY: runtime.upstreamBearerToken,
    ANTHROPIC_MODEL: runtime.forcedModel,
    ANTHROPIC_SMALL_FAST_MODEL: runtime.forcedModel,
    ANTHROPIC_DEFAULT_OPUS_MODEL: runtime.forcedModel,
    ANTHROPIC_DEFAULT_SONNET_MODEL: runtime.forcedModel,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: runtime.forcedModel,
    CLAUDE_CODE_SUBAGENT_MODEL: runtime.forcedModel,
  };
  if (safeMode) {
    childEnv.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1";
  } else {
    delete childEnv.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC;
  }

  const child = spawn(claudeBinary, injectedArgs, {
    stdio: "inherit",
    env: childEnv,
  });

  const forwardSignal = (signal: NodeJS.Signals): void => {
    if (child.exitCode === null && !child.killed) {
      child.kill(signal);
    }
  };
  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));

  const exitCode = await new Promise<number>((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (signal) {
        resolve(128);
        return;
      }
      resolve(code ?? 0);
    });
  });

  await new Promise<void>((resolve) => {
    proxyServer.close(() => resolve());
  });

  process.exit(exitCode);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`claudex: ${message}`);
  process.exit(1);
});
