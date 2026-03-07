import { describe, expect, test } from "bun:test";
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
} from "../src/core.ts";

describe("hasEffortFlag", () => {
  test("detects --effort and --effort=", () => {
    expect(hasEffortFlag(["--foo", "--effort"])).toBe(true);
    expect(hasEffortFlag(["--effort=xhigh"])).toBe(true);
    expect(hasEffortFlag(["--model", "x"])).toBe(false);
  });
});

describe("parseClaudexArgs", () => {
  test("default safe mode is true", () => {
    const parsed = parseClaudexArgs(["-p", "hello"]);
    expect(parsed.safeMode).toBe(true);
    expect(parsed.hasSettingsArg).toBe(false);
    expect(parsed.claudeArgs).toEqual(["-p", "hello"]);
  });

  test("consumes --no-safe and disables safe mode", () => {
    const parsed = parseClaudexArgs(["--no-safe", "-p", "hello"]);
    expect(parsed.safeMode).toBe(false);
    expect(parsed.hasSettingsArg).toBe(false);
    expect(parsed.claudeArgs).toEqual(["-p", "hello"]);
  });

  test("detects --settings argument", () => {
    expect(parseClaudexArgs(["--settings", "{\"a\":1}"]).hasSettingsArg).toBe(true);
    expect(parseClaudexArgs(["--settings={\"a\":1}"]).hasSettingsArg).toBe(true);
  });
});

describe("parseCodexConfig", () => {
  const configToml = `
model_provider = "unlimitex"
model = "gpt-5.3-codex"

[model_providers.voids]
name = "voids"
base_url = "https://voids.example/v1"
wire_api = "responses"

[model_providers.unlimitex]
name = "unlimitex"
base_url = "https://unlimitex.example/v1"
wire_api = "responses"
`;

  test("parses model and providers", () => {
    const parsed = parseCodexConfig(configToml);
    expect(parsed.modelProvider).toBe("unlimitex");
    expect(parsed.model).toBe("gpt-5.3-codex");
    expect(parsed.providers.unlimitex.baseUrl).toBe("https://unlimitex.example/v1");
  });

  test("resolves selected provider base url", () => {
    const resolved = resolveUpstreamFromCodexConfig(configToml);
    expect(resolved.baseUrl).toBe("https://unlimitex.example/v1");
    expect(resolved.model).toBe("gpt-5.3-codex");
  });

  test("base url override wins", () => {
    const resolved = resolveUpstreamFromCodexConfig(configToml, {
      baseUrlOverride: "https://override.example/v1",
    });
    expect(resolved.baseUrl).toBe("https://override.example/v1");
  });
});

describe("parseApiKeyFromAuthJson", () => {
  test("reads OPENAI_API_KEY", () => {
    const authJson = JSON.stringify({ OPENAI_API_KEY: "sk-test" });
    expect(parseApiKeyFromAuthJson(authJson)).toBe("sk-test");
  });

  test("env override wins", () => {
    const authJson = JSON.stringify({ OPENAI_API_KEY: "sk-file" });
    expect(parseApiKeyFromAuthJson(authJson, "sk-env")).toBe("sk-env");
  });

  test("throws without key", () => {
    const authJson = JSON.stringify({ tokens: { access_token: "x" } });
    expect(() => parseApiKeyFromAuthJson(authJson)).toThrow("failed to read OPENAI API key");
  });
});

describe("parseChatgptTokenFromAuthJson", () => {
  test("prefers tokens.access_token and reads account_id", () => {
    const authJson = JSON.stringify({
      tokens: {
        id_token: "id-token-value",
        access_token: "access-token-value",
        account_id: "acct_123",
      },
    });
    const parsed = parseChatgptTokenFromAuthJson(authJson);
    expect(parsed.bearerToken).toBe("access-token-value");
    expect(parsed.accountId).toBe("acct_123");
    expect(parsed.source).toBe("tokens.access_token");
  });

  test("falls back to tokens.id_token", () => {
    const authJson = JSON.stringify({
      tokens: {
        id_token: "id-token-value",
      },
    });
    const parsed = parseChatgptTokenFromAuthJson(authJson);
    expect(parsed.bearerToken).toBe("id-token-value");
    expect(parsed.source).toBe("tokens.id_token");
  });

  test("env bearer token wins", () => {
    const authJson = JSON.stringify({
      tokens: {
        id_token: "id-token-value",
      },
    });
    const parsed = parseChatgptTokenFromAuthJson(authJson, {
      envBearerToken: "env-token-value",
      envAccountId: "acct_env",
    });
    expect(parsed.bearerToken).toBe("env-token-value");
    expect(parsed.accountId).toBe("acct_env");
    expect(parsed.source).toBe("env");
  });

  test("throws without token fields", () => {
    const authJson = JSON.stringify({ OPENAI_API_KEY: "sk-test" });
    expect(() => parseChatgptTokenFromAuthJson(authJson)).toThrow("failed to read ChatGPT token");
  });
});

describe("parseChatgptRefreshConfigFromAuthJson", () => {
  test("reads refresh token and client id from id_token aud", () => {
    const payload = Buffer.from(JSON.stringify({ aud: ["app_test_client_id"] })).toString("base64url");
    const idToken = `x.${payload}.y`;
    const authJson = JSON.stringify({
      tokens: {
        id_token: idToken,
        refresh_token: "refresh-token-value",
      },
    });
    const parsed = parseChatgptRefreshConfigFromAuthJson(authJson);
    expect(parsed.refreshToken).toBe("refresh-token-value");
    expect(parsed.clientId).toBe("app_test_client_id");
  });
});

describe("approxTokenCount", () => {
  test("counts text parts", () => {
    const count = approxTokenCount({
      messages: [{ content: "abcd" }, { content: [{ text: "1234" }, { content: "abcd" }] }],
    });
    expect(count).toBeGreaterThanOrEqual(3);
  });

  test("returns at least one", () => {
    expect(approxTokenCount({})).toBe(1);
  });
});

describe("applyDefaultEffort", () => {
  test("sets xhigh for gpt-5.3-codex by default", () => {
    const body: Record<string, any> = {};
    applyDefaultEffort(body, {
      forcedModel: "gpt-5.3-codex",
      defaultReasoningEffort: "xhigh",
      preserveClientEffort: false,
    });
    expect(body.output_config.effort).toBe("xhigh");
    expect(body.reasoning.effort).toBe("xhigh");
  });

  test("does not overwrite when preserving client effort", () => {
    const body: Record<string, any> = {};
    applyDefaultEffort(body, {
      forcedModel: "gpt-5.3-codex",
      defaultReasoningEffort: "xhigh",
      preserveClientEffort: true,
    });
    expect(body.output_config).toBeUndefined();
  });
});

describe("sanitizeToolFields", () => {
  test("removes defer_loading from each tool", () => {
    const body: Record<string, any> = {
      tools: [{ name: "a", defer_loading: true }, { name: "b" }],
    };
    const removed = sanitizeToolFields(body);
    expect(removed).toBe(1);
    expect(body.tools[0].defer_loading).toBeUndefined();
  });
});

describe("extractInstructionsFromSystem", () => {
  test("joins string and object text blocks", () => {
    const text = extractInstructionsFromSystem(["alpha", { type: "text", text: "beta" }]);
    expect(text).toBe("alpha\n\nbeta");
  });
});

describe("toResponsesInput", () => {
  test("maps text, tool_use and tool_result parts", () => {
    const input = toResponsesInput([
      {
        role: "assistant",
        content: [
          { type: "text", text: "Scanning files" },
          { type: "tool_use", id: "toolu_1", name: "list_files", input: { path: "." } },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "README.md\nsrc/" }],
      },
    ]);

    expect(input[0]).toEqual({
      role: "assistant",
      content: [{ type: "output_text", text: "Scanning files" }],
    });
    expect(input[1]).toEqual({
      type: "function_call",
      call_id: "toolu_1",
      name: "list_files",
      arguments: JSON.stringify({ path: "." }),
    });
    expect(input[2]).toEqual({
      type: "function_call_output",
      call_id: "toolu_1",
      output: "README.md\nsrc/",
    });
  });
});

describe("mapAnthropicToolsToResponsesTools", () => {
  test("maps anthropic tool schema to responses function tool", () => {
    const tools = mapAnthropicToolsToResponsesTools([
      {
        name: "read_file",
        description: "Read file",
        input_schema: {
          type: "object",
          properties: {
            path: { type: "string" },
          },
          required: ["path"],
        },
      },
    ]);

    expect(tools).toEqual([
      {
        type: "function",
        name: "read_file",
        description: "Read file",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string" },
          },
          required: ["path"],
        },
      },
    ]);
  });
});

describe("mapAnthropicToolChoiceToResponsesToolChoice", () => {
  test("maps type:any to required", () => {
    expect(mapAnthropicToolChoiceToResponsesToolChoice({ type: "any" })).toBe("required");
  });

  test("maps explicit tool name to function choice", () => {
    expect(mapAnthropicToolChoiceToResponsesToolChoice({ type: "tool", name: "read_file" })).toEqual({
      type: "function",
      name: "read_file",
    });
  });
});

describe("mapResponsesOutputToAnthropicContent", () => {
  test("maps message output_text to anthropic text content", () => {
    const mapped = mapResponsesOutputToAnthropicContent([
      {
        type: "message",
        content: [{ type: "output_text", text: "Project has src and tests directories." }],
      },
    ]);
    expect(mapped.stopReason).toBe("end_turn");
    expect(mapped.content).toEqual([
      {
        type: "text",
        text: "Project has src and tests directories.",
      },
    ]);
  });

  test("maps function_call to anthropic tool_use and tool_use stop reason", () => {
    const mapped = mapResponsesOutputToAnthropicContent([
      {
        type: "function_call",
        call_id: "call_1",
        name: "read_file",
        arguments: "{\"path\":\"README.md\"}",
      },
    ]);
    expect(mapped.stopReason).toBe("tool_use");
    expect(mapped.content).toEqual([
      {
        type: "tool_use",
        id: "call_1",
        name: "read_file",
        input: { path: "README.md" },
      },
    ]);
  });
});
