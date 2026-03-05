export type JsonObject = Record<string, any>;

export function hasEffortFlag(args: string[]): boolean {
  return args.some((arg) => arg === "--effort" || arg.startsWith("--effort="));
}

export function parseClaudexArgs(rawArgs: string[]): {
  claudeArgs: string[];
  safeMode: boolean;
  hasSettingsArg: boolean;
} {
  let safeMode = true;
  let hasSettingsArg = false;
  const claudeArgs: string[] = [];

  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];

    if (arg === "--no-safe") {
      safeMode = false;
      continue;
    }
    if (arg === "--settings" || arg.startsWith("--settings=")) {
      hasSettingsArg = true;
    }
    claudeArgs.push(arg);
  }

  return { claudeArgs, safeMode, hasSettingsArg };
}

export interface ParsedCodexConfig {
  model?: string;
  modelProvider?: string;
  providers: Record<
    string,
    {
      key: string;
      name?: string;
      baseUrl?: string;
      wireApi?: string;
    }
  >;
}

function parseTopLevelString(contents: string, key: string): string | undefined {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = contents.match(new RegExp(`^\\s*${escaped}\\s*=\\s*"([^"]+)"`, "m"));
  return match?.[1]?.trim();
}

export function parseCodexConfig(contents: string): ParsedCodexConfig {
  const providers: ParsedCodexConfig["providers"] = {};
  const headerRegex = /^\[model_providers\.([^\]]+)\]\s*$/gm;

  const headers = Array.from(contents.matchAll(headerRegex));
  for (let i = 0; i < headers.length; i += 1) {
    const current = headers[i];
    const next = headers[i + 1];

    const providerKey = current[1]?.trim();
    if (!providerKey) {
      continue;
    }

    const blockStart = (current.index ?? 0) + current[0].length;
    const blockEnd = next?.index ?? contents.length;
    const block = contents.slice(blockStart, blockEnd);

    providers[providerKey] = {
      key: providerKey,
      name: parseTopLevelString(block, "name"),
      baseUrl: parseTopLevelString(block, "base_url"),
      wireApi: parseTopLevelString(block, "wire_api"),
    };
  }

  return {
    model: parseTopLevelString(contents, "model"),
    modelProvider: parseTopLevelString(contents, "model_provider"),
    providers,
  };
}

export function resolveUpstreamFromCodexConfig(
  contents: string,
  options: {
    providerOverride?: string;
    baseUrlOverride?: string;
  } = {}
): { baseUrl: string; providerKey?: string; model?: string } {
  const parsed = parseCodexConfig(contents);

  if (options.baseUrlOverride && options.baseUrlOverride.trim()) {
    return {
      baseUrl: options.baseUrlOverride.trim(),
      providerKey: options.providerOverride || parsed.modelProvider,
      model: parsed.model,
    };
  }

  const preferredProvider = options.providerOverride || parsed.modelProvider;
  if (preferredProvider) {
    const chosen = parsed.providers[preferredProvider];
    if (chosen?.baseUrl?.trim()) {
      return {
        baseUrl: chosen.baseUrl.trim(),
        providerKey: preferredProvider,
        model: parsed.model,
      };
    }
  }

  for (const provider of Object.values(parsed.providers)) {
    if (provider.baseUrl?.trim()) {
      return {
        baseUrl: provider.baseUrl.trim(),
        providerKey: provider.key,
        model: parsed.model,
      };
    }
  }

  throw new Error("failed to resolve base_url from ~/.codex/config.toml");
}

function parseAuthJson(contents: string): any {
  try {
    return JSON.parse(contents);
  } catch {
    throw new Error("failed to parse ~/.codex/auth.json as JSON");
  }
}

function firstNonEmptyString(candidates: unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  return undefined;
}

export function parseApiKeyFromAuthJson(contents: string, envApiKey?: string): string {
  if (envApiKey?.trim()) {
    return envApiKey.trim();
  }

  const parsed = parseAuthJson(contents);

  const apiKey = firstNonEmptyString([
    parsed?.OPENAI_API_KEY,
    parsed?.openai_api_key,
    parsed?.api_key,
    parsed?.openai?.api_key,
    parsed?.providers?.openai?.api_key,
  ]);
  if (apiKey) {
    return apiKey;
  }

  throw new Error("failed to read OPENAI API key from ~/.codex/auth.json");
}

export interface ParsedChatgptTokenFromAuth {
  bearerToken: string;
  accountId?: string;
  source: "env" | "tokens.id_token" | "tokens.access_token" | "id_token" | "access_token";
}

export function parseChatgptTokenFromAuthJson(
  contents: string,
  options: {
    envBearerToken?: string;
    envAccountId?: string;
  } = {}
): ParsedChatgptTokenFromAuth {
  if (options.envBearerToken?.trim()) {
    let parsed: any = {};
    if (contents.trim().length > 0) {
      parsed = parseAuthJson(contents);
    }
    const accountId = firstNonEmptyString([
      options.envAccountId,
      parsed?.tokens?.account_id,
      parsed?.account_id,
      parsed?.chatgpt_account_id,
      parsed?.chatgptAccountId,
    ]);
    return {
      bearerToken: options.envBearerToken.trim(),
      accountId,
      source: "env",
    };
  }

  const parsed = parseAuthJson(contents);
  const accountId = firstNonEmptyString([
    options.envAccountId,
    parsed?.tokens?.account_id,
    parsed?.account_id,
    parsed?.chatgpt_account_id,
    parsed?.chatgptAccountId,
  ]);

  const orderedCandidates: Array<{
    value: unknown;
    source: ParsedChatgptTokenFromAuth["source"];
  }> = [
    { value: parsed?.tokens?.access_token, source: "tokens.access_token" },
    { value: parsed?.tokens?.id_token, source: "tokens.id_token" },
    { value: parsed?.access_token, source: "access_token" },
    { value: parsed?.id_token, source: "id_token" },
  ];

  for (const candidate of orderedCandidates) {
    if (typeof candidate.value === "string" && candidate.value.trim().length > 0) {
      return {
        bearerToken: candidate.value.trim(),
        accountId,
        source: candidate.source,
      };
    }
  }

  throw new Error(
    "failed to read ChatGPT token from ~/.codex/auth.json (expected tokens.id_token or tokens.access_token)"
  );
}

function decodeJwtPayload(token?: string): any | null {
  if (typeof token !== "string") {
    return null;
  }
  const parts = token.split(".");
  if (parts.length < 2) {
    return null;
  }

  try {
    const normalized = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(Buffer.from(normalized, "base64").toString("utf8"));
  } catch {
    return null;
  }
}

export interface ParsedChatgptRefreshConfigFromAuth {
  refreshToken?: string;
  clientId?: string;
}

export function parseChatgptRefreshConfigFromAuthJson(contents: string): ParsedChatgptRefreshConfigFromAuth {
  const parsed = parseAuthJson(contents);

  const refreshToken = firstNonEmptyString([parsed?.tokens?.refresh_token, parsed?.refresh_token]);
  let clientId = firstNonEmptyString([
    parsed?.tokens?.client_id,
    parsed?.client_id,
    parsed?.oauth?.client_id,
  ]);

  if (!clientId) {
    const idToken = firstNonEmptyString([parsed?.tokens?.id_token, parsed?.id_token]);
    const payload = decodeJwtPayload(idToken);
    const aud = payload?.aud;
    if (typeof aud === "string" && aud.trim().length > 0) {
      clientId = aud.trim();
    } else if (Array.isArray(aud) && typeof aud[0] === "string" && aud[0].trim().length > 0) {
      clientId = aud[0].trim();
    }
  }

  return { refreshToken, clientId };
}

export function approxTokenCount(body: JsonObject): number {
  const lines: string[] = [];
  if (Array.isArray(body?.messages)) {
    for (const message of body.messages) {
      if (typeof message?.content === "string") {
        lines.push(message.content);
        continue;
      }
      if (Array.isArray(message?.content)) {
        for (const part of message.content) {
          if (typeof part?.text === "string") {
            lines.push(part.text);
          }
          if (typeof part?.content === "string") {
            lines.push(part.content);
          }
        }
      }
    }
  }

  const text = lines.join("\n");
  return Math.max(1, Math.ceil(text.length / 4));
}

export function hasExplicitEffort(body: JsonObject): boolean {
  return Boolean(
    (typeof body?.effort === "string" && body.effort.length > 0) ||
      (typeof body?.output_config?.effort === "string" && body.output_config.effort.length > 0) ||
      (typeof body?.reasoning?.effort === "string" && body.reasoning.effort.length > 0)
  );
}

export function applyDefaultEffort(
  body: JsonObject,
  options: {
    forcedModel: string;
    defaultReasoningEffort: string;
    preserveClientEffort: boolean;
  }
): void {
  if (options.forcedModel !== "gpt-5.3-codex") {
    return;
  }
  if (options.preserveClientEffort || hasExplicitEffort(body)) {
    return;
  }

  if (typeof body.output_config !== "object" || body.output_config === null) {
    body.output_config = {};
  }
  body.output_config.effort = options.defaultReasoningEffort;

  if (typeof body.reasoning !== "object" || body.reasoning === null) {
    body.reasoning = {};
  }
  body.reasoning.effort = options.defaultReasoningEffort;
}

export function sanitizeToolFields(body: JsonObject): number {
  let removed = 0;
  if (!Array.isArray(body?.tools)) {
    return removed;
  }

  for (const tool of body.tools) {
    if (!tool || typeof tool !== "object") {
      continue;
    }
    if ("defer_loading" in tool) {
      delete tool.defer_loading;
      removed += 1;
    }
  }

  return removed;
}
