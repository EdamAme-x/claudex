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
  extractInstructionsFromSystem,
  hasEffortFlag,
  mapAnthropicToolChoiceToResponsesToolChoice,
  mapAnthropicToolsToResponsesTools,
  mapResponsesOutputToAnthropicContent,
  parseChatgptRefreshConfigFromAuthJson,
  parseChatgptTokenFromAuthJson,
  parseClaudexArgs,
  parseApiKeyFromAuthJson,
  parseCodexConfig,
  resolveUpstreamFromCodexConfig,
  sanitizeToolFields,
  toResponsesInput,
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
  safeMode: boolean;
  workspaceSummary?: string;
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

function buildWorkspaceSummary(rootDir: string): string | undefined {
  try {
    const ignored = new Set([
      ".git",
      "node_modules",
      "dist",
      ".DS_Store",
      ".idea",
      ".vscode",
    ]);
    const topEntries = readdirSync(rootDir, { withFileTypes: true })
      .filter((entry) => !ignored.has(entry.name))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));

    const dirs = topEntries.filter((entry) => entry.isDirectory()).map((entry) => entry.name);
    const files = topEntries.filter((entry) => entry.isFile()).map((entry) => entry.name);
    const lines: string[] = [];
    lines.push(`cwd: ${rootDir}`);
    lines.push(`top-level dirs: ${dirs.slice(0, 20).join(", ") || "(none)"}`);
    lines.push(`top-level files: ${files.slice(0, 30).join(", ") || "(none)"}`);

    const inspectDirs = ["src", "tests", "scripts"];
    for (const dirName of inspectDirs) {
      const dirPath = join(rootDir, dirName);
      if (!existsSync(dirPath)) {
        continue;
      }
      try {
        const children = readdirSync(dirPath, { withFileTypes: true })
          .filter((entry) => entry.isFile())
          .map((entry) => entry.name)
          .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
        lines.push(`${dirName}/ files: ${children.slice(0, 30).join(", ") || "(none)"}`);
      } catch {
        // ignore directory scan errors
      }
    }

    const readmePath = join(rootDir, "README.md");
    if (existsSync(readmePath)) {
      try {
        const readmePreview = readFileSync(readmePath, "utf8").split("\n").slice(0, 24).join("\n");
        if (readmePreview.trim().length > 0) {
          lines.push("README.md preview:");
          lines.push(readmePreview);
        }
      } catch {
        // ignore README read errors
      }
    }

    return lines.join("\n");
  } catch {
    return undefined;
  }
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

  const forceModelFromEnv = process.env.CLAUDEX_FORCE_MODEL?.trim();
  const forceModelFromConfig = modelFromConfig?.trim();
  let forcedModelSource: "env" | "config" | "default" = "default";
  if (forceModelFromEnv && forceModelFromEnv.length > 0) {
    forcedModelSource = "env";
  } else if (forceModelFromConfig && forceModelFromConfig.length > 0) {
    forcedModelSource = "config";
  }
  const defaultForcedModel = "gpt-5.3-codex";
  const forcedModel = (forceModelFromEnv || forceModelFromConfig || defaultForcedModel).trim();

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

    // In ChatGPT mode, gpt-5.3-codex is not always available by default.
    // Keep user-configured model values, but switch the default to a safer baseline.
    let chatgptForcedModel = forcedModel;
    if (forcedModelSource === "default" && forcedModel === defaultForcedModel) {
      chatgptForcedModel = (process.env.CLAUDEX_CHATGPT_DEFAULT_MODEL || "gpt-5-codex").trim();
    }

    return {
      upstreamBaseUrl: chatgptBaseUrl,
      upstreamBearerToken: tokenAuth.bearerToken,
      upstreamExtraHeaders: extraHeaders,
      forcedModel: chatgptForcedModel,
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
    let chatgptForcedModel = forcedModel;
    if (forcedModelSource === "default" && forcedModel === defaultForcedModel) {
      chatgptForcedModel = (process.env.CLAUDEX_CHATGPT_DEFAULT_MODEL || "gpt-5-codex").trim();
    }
    return {
      upstreamBaseUrl: chatgptBaseUrl,
      upstreamBearerToken,
      upstreamExtraHeaders: {},
      forcedModel: chatgptForcedModel,
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
    const normalized = key.toLowerCase();
    if (
      normalized === "transfer-encoding" ||
      normalized === "content-encoding" ||
      normalized === "content-length"
    ) {
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

function rewriteRequestPathForChatgptCodex(upstreamOrigin: URL, requestPath: string): string {
  const incoming = new URL(requestPath, "http://localhost");
  const basePath = normalizeBasePath(upstreamOrigin.pathname);
  const isChatgptCodex =
    upstreamOrigin.hostname === "chatgpt.com" && basePath.startsWith("/backend-api/codex");

  if (isChatgptCodex && incoming.pathname === "/v1/messages") {
    incoming.pathname = "/responses";
  }

  return `${incoming.pathname}${incoming.search}`;
}

function isChatgptCodexEndpoint(upstreamOrigin: URL): boolean {
  const basePath = normalizeBasePath(upstreamOrigin.pathname);
  return upstreamOrigin.hostname === "chatgpt.com" && basePath.startsWith("/backend-api/codex");
}

function writeSseEvent(res: http.ServerResponse, event: string, data: unknown): void {
  res.write(`event: ${event}\n`);
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function buildForwardHeaders(
  req: http.IncomingMessage,
  authState: UpstreamAuthState,
  contentLength: number
): Record<string, string> {
  const headers: Record<string, string> = {
    authorization: `Bearer ${authState.bearerToken}`,
    "content-type": "application/json",
    "content-length": String(contentLength),
    "accept-encoding": "identity",
  };

  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) {
      continue;
    }
    const normalized = key.toLowerCase();
    if (
      normalized === "host" ||
      normalized === "content-length" ||
      normalized === "authorization" ||
      normalized === "accept-encoding"
    ) {
      continue;
    }
    headers[normalized] = Array.isArray(value) ? value.join(", ") : value;
  }

  for (const [key, value] of Object.entries(authState.extraHeaders)) {
    headers[key.toLowerCase()] = value;
  }

  return headers;
}

async function proxyChatgptCodexResponsesAsAnthropic(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  requestPath: string,
  upstreamOrigin: URL,
  authState: UpstreamAuthState,
  overrideBody: JsonObject,
  debugLogging: boolean
): Promise<void> {
  const rewrittenPath = rewriteRequestPathForChatgptCodex(upstreamOrigin, requestPath);
  const upstreamUrl = buildUpstreamUrl(upstreamOrigin, rewrittenPath);

  const upstreamPayload = {
    ...overrideBody,
    stream: true,
  };
  const upstreamPayloadString = JSON.stringify(upstreamPayload);
  const headers = buildForwardHeaders(req, authState, Buffer.byteLength(upstreamPayloadString));

  if (debugLogging) {
    console.error(`claudex-proxy upstream request: ${req.method || "POST"} ${upstreamUrl.toString()}`);
  }

  const upstreamResponse = await fetch(upstreamUrl, {
    method: "POST",
    headers,
    body: upstreamPayloadString,
  });

  const upstreamContentType = upstreamResponse.headers.get("content-type") || "unknown";
  if (debugLogging) {
    console.error(
      `claudex-proxy upstream response: status=${upstreamResponse.status} url=${upstreamUrl.toString()} content_type=${upstreamContentType}`
    );
  }

  if (upstreamResponse.status >= 400) {
    const errorText = await upstreamResponse.text();
    if (debugLogging) {
      console.error(
        `claudex-proxy upstream error: status=${upstreamResponse.status} url=${upstreamUrl.toString()} body=${errorText.slice(0, 300)}`
      );
    }
    res.writeHead(upstreamResponse.status, { "content-type": "application/json" });
    res.end(errorText);
    return;
  }

  if (!upstreamResponse.body) {
    writeJson(res, 502, {
      type: "error",
      error: {
        type: "api_error",
        message: "claudex-proxy received empty stream body from ChatGPT responses endpoint",
      },
    });
    return;
  }

  const streamRequested = overrideBody.stream !== false;
  const responseId = `msg_${Date.now().toString(16)}`;
  let model = String(overrideBody.model || "");
  let text = "";
  let parsedContent: Array<Record<string, unknown>> = [];
  let stopReason: "tool_use" | "end_turn" = "end_turn";
  let usageInputTokens = 0;
  let usageOutputTokens = 0;

  const decoder = new TextDecoder();
  let buffered = "";
  const readable = Readable.fromWeb(upstreamResponse.body as any);
  for await (const chunk of readable) {
    buffered += decoder.decode(chunk as Buffer, { stream: true });

    let boundary = buffered.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffered.slice(0, boundary);
      buffered = buffered.slice(boundary + 2);

      let eventName = "";
      let dataText = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) {
          eventName = line.slice("event:".length).trim();
        } else if (line.startsWith("data:")) {
          dataText += line.slice("data:".length).trim();
        }
      }

      if (!eventName || !dataText) {
        boundary = buffered.indexOf("\n\n");
        continue;
      }

      let parsed: any;
      try {
        parsed = JSON.parse(dataText);
      } catch {
        boundary = buffered.indexOf("\n\n");
        continue;
      }

      if (parsed?.type === "response.created" && parsed?.response?.model) {
        model = String(parsed.response.model);
      }
      if (parsed?.type === "response.output_text.delta" && typeof parsed.delta === "string") {
        text += parsed.delta;
      }

      if (parsed?.type === "response.completed") {
        const completedUsage = parsed?.response?.usage;
        usageInputTokens = Number(completedUsage?.input_tokens || 0);
        usageOutputTokens = Number(completedUsage?.output_tokens || 0);
        if (typeof parsed?.response?.model === "string" && parsed.response.model.length > 0) {
          model = parsed.response.model;
        }
        if (Array.isArray(parsed?.response?.output)) {
          if (debugLogging) {
            const outputTypes = parsed.response.output
              .map((item: any) => (item && typeof item.type === "string" ? item.type : "unknown"))
              .join(",");
            console.error(`claudex-proxy responses output types: [${outputTypes}]`);
          }
          const mapped = mapResponsesOutputToAnthropicContent(parsed.response.output);
          parsedContent = mapped.content;
          stopReason = mapped.stopReason;
        }
      }

      boundary = buffered.indexOf("\n\n");
    }
  }

  if (parsedContent.length === 0 && text.length > 0) {
    parsedContent = [{ type: "text", text }];
  }
  if (debugLogging) {
    const contentKinds = parsedContent.map((block) => String(block.type || "unknown")).join(",");
    const toolUseSummaries = parsedContent
      .filter((block) => block.type === "tool_use")
      .map((block) => {
        const name = typeof block.name === "string" ? block.name : "unknown";
        const input = block.input;
        const keys =
          input && typeof input === "object" && !Array.isArray(input) ? Object.keys(input).join("|") : "";
        return `${name}(${keys})`;
      })
      .join(",");
    console.error(
      `claudex-proxy mapped response summary: blocks=${parsedContent.length}, stop_reason=${stopReason}, content_types=[${contentKinds}], tool_uses=[${toolUseSummaries}], fallback_text_len=${text.length}`
    );
  }

  if (streamRequested) {
    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    writeSseEvent(res, "message_start", {
      type: "message_start",
      message: {
        id: responseId,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });

    for (let index = 0; index < parsedContent.length; index += 1) {
      const block = parsedContent[index];
      if (block.type === "text") {
        writeSseEvent(res, "content_block_start", {
          type: "content_block_start",
          index,
          content_block: {
            type: "text",
            text: "",
          },
        });
        writeSseEvent(res, "content_block_delta", {
          type: "content_block_delta",
          index,
          delta: {
            type: "text_delta",
            text: typeof block.text === "string" ? block.text : "",
          },
        });
        writeSseEvent(res, "content_block_stop", {
          type: "content_block_stop",
          index,
        });
        continue;
      }

      if (block.type === "tool_use") {
        writeSseEvent(res, "content_block_start", {
          type: "content_block_start",
          index,
          content_block: block,
        });
        writeSseEvent(res, "content_block_stop", {
          type: "content_block_stop",
          index,
        });
      }
    }

    writeSseEvent(res, "message_delta", {
      type: "message_delta",
      delta: {
        stop_reason: stopReason,
        stop_sequence: null,
      },
      usage: {
        output_tokens: usageOutputTokens,
      },
    });
    writeSseEvent(res, "message_stop", { type: "message_stop" });
    res.end();
    return;
  }

  writeJson(res, 200, {
    id: responseId,
    type: "message",
    role: "assistant",
    model,
    content: parsedContent,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: usageInputTokens,
      output_tokens: usageOutputTokens,
    },
  });
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
    if (
      normalized === "host" ||
      normalized === "content-length" ||
      normalized === "authorization" ||
      normalized === "accept-encoding"
    ) {
      continue;
    }
    headers[normalized] = Array.isArray(value) ? value.join(", ") : value;
  }
  headers["accept-encoding"] = "identity";
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

  const rewrittenPath = rewriteRequestPathForChatgptCodex(upstreamOrigin, requestPath);
  const upstreamUrl = buildUpstreamUrl(upstreamOrigin, rewrittenPath);
  if (debugLogging) {
    console.error(`claudex-proxy upstream request: ${req.method || "GET"} ${upstreamUrl.toString()}`);
  }
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

  const copiedHeaders = copyHeadersFromUpstream(upstreamResponse.headers);
  if (debugLogging) {
    const contentType = upstreamResponse.headers.get("content-type") || "unknown";
    console.error(
      `claudex-proxy upstream response: status=${upstreamResponse.status} url=${upstreamUrl.toString()} content_type=${contentType}`
    );
  }
  if (debugLogging && upstreamResponse.status >= 400) {
    const bodyPreview = await upstreamResponse.text();
    console.error(
      `claudex-proxy upstream error: status=${upstreamResponse.status} url=${upstreamUrl.toString()} body=${bodyPreview.slice(0, 300)}`
    );
    res.writeHead(upstreamResponse.status, copiedHeaders);
    res.end(bodyPreview);
    return;
  }

  res.writeHead(upstreamResponse.status, copiedHeaders);
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
        if (isChatgptCodexEndpoint(upstreamOrigin) && typeof parsed.instructions !== "string") {
          const inferredInstructions = extractInstructionsFromSystem(parsed.system);
          if (inferredInstructions && inferredInstructions.length > 0) {
            parsed.instructions = inferredInstructions;
          }
        }
        if (isChatgptCodexEndpoint(upstreamOrigin)) {
          parsed.input = toResponsesInput(parsed.messages);
          if (options.debug && Array.isArray(parsed.input)) {
            const inputItemKinds = parsed.input
              .map((item: any) => (item && typeof item.type === "string" ? item.type : "message"))
              .slice(0, 8);
            const functionCallOutputCount = parsed.input.filter(
              (item: any) => item && item.type === "function_call_output"
            ).length;
            console.error(
              `claudex-proxy responses input summary: items=${parsed.input.length}, first_types=[${inputItemKinds.join(",")}], function_call_output_count=${functionCallOutputCount}`
            );
          }
          const mappedTools = mapAnthropicToolsToResponsesTools(parsed.tools);
          const filteredMappedTools =
            options.safeMode
              ? mappedTools.filter((tool) => String(tool.name || "").trim().toLowerCase() !== "toolsearch")
              : mappedTools;
          if (options.debug) {
            const mappedToolNames = filteredMappedTools
              .map((tool) => String(tool.name || "").trim())
              .filter((name) => name.length > 0)
              .slice(0, 10)
              .join(",");
            console.error(
              `claudex-proxy tool map summary: original=${mappedTools.length}, filtered=${filteredMappedTools.length}, safe_mode=${options.safeMode}, first_tools=[${mappedToolNames}]`
            );
          }
          if (filteredMappedTools.length === 0 && options.workspaceSummary && options.workspaceSummary.length > 0) {
            const existingInstructions = typeof parsed.instructions === "string" ? parsed.instructions : "";
            parsed.instructions = `${existingInstructions}\n\nLocal workspace snapshot (autogenerated):\n${options.workspaceSummary}\n\nWhen asked about this folder/project, answer concretely and immediately from this snapshot. Do not say you will inspect or check files. Do not ask for extra time before answering.`.trim();
            if (options.debug) {
              console.error("claudex-proxy injected workspace summary into instructions (no tools available)");
            }
          }
          if (filteredMappedTools.length > 0) {
            parsed.tools = filteredMappedTools;
          } else {
            delete parsed.tools;
          }
          const mappedToolChoice = mapAnthropicToolChoiceToResponsesToolChoice(parsed.tool_choice);
          if (mappedToolChoice !== undefined) {
            parsed.tool_choice = mappedToolChoice;
          } else {
            delete parsed.tool_choice;
          }
          const inferredEffort = parsed.output_config?.effort ?? parsed.reasoning?.effort;
          if (typeof inferredEffort === "string" && inferredEffort.length > 0) {
            parsed.reasoning = {
              ...(parsed.reasoning && typeof parsed.reasoning === "object" ? parsed.reasoning : {}),
              effort: inferredEffort,
            };
          }
          delete parsed.messages;
          delete parsed.system;
          delete parsed.max_tokens;
          delete parsed.max_output_tokens;
          delete parsed.output_config;
          delete parsed.thinking;
          delete parsed.metadata;
          parsed.store = false;
        }

        if (options.debug) {
          const effort = parsed.output_config?.effort ?? parsed.reasoning?.effort ?? "unset";
          const payloadKeys = Object.keys(parsed).join(",");
          const firstMessagePreview =
            Array.isArray(parsed.messages) && parsed.messages.length > 0
              ? JSON.stringify(parsed.messages[0]).slice(0, 300)
              : "none";
          console.error(
            `claudex-proxy model remap: ${originalModel} -> ${options.forcedModel}, effort=${effort}, preserve_client_effort=${options.preserveClientEffort}, removed_tool_fields=${removedToolFields}, payload_keys=[${payloadKeys}], first_message=${firstMessagePreview}`
          );
        }

        if (isChatgptCodexEndpoint(upstreamOrigin)) {
          await proxyChatgptCodexResponsesAsAnthropic(
            req,
            res,
            url.pathname + url.search,
            upstreamOrigin,
            authState,
            parsed,
            options.debug
          );
          return;
        }

        await proxyRaw(req, res, bodyBuffer, url.pathname + url.search, upstreamOrigin, authState, options.debug, parsed);
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
  const workspaceSummary = buildWorkspaceSummary(process.cwd());
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
      safeMode,
      workspaceSummary,
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
