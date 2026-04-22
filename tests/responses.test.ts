import { describe, expect, test } from "bun:test";
import {
  mapResponsesOutputToAnthropicContent,
  sanitizeToolFields,
  toResponsesInput,
} from "../src/anthropic-responses.ts";
import { adaptAnthropicMessagesRequestForResponses } from "../src/proxy.ts";
import { mapAnthropicToolsToResponsesTools } from "../src/tool-schema.ts";
import { rewriteRequestPath } from "../src/upstream.ts";

describe("responses bridge", () => {
  test("maps anthropic messages with tool events into Responses input", () => {
    const mapped = toResponsesInput([
      { role: "user", content: [{ type: "text", text: "hello" }] },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu_1", name: "grep", input: { pattern: "foo" } }],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu_1", content: [{ type: "text", text: "bar" }] }],
      },
    ]);

    expect(mapped).toEqual([
      {
        role: "user",
        content: [{ type: "input_text", text: "hello" }],
      },
      {
        type: "function_call",
        call_id: "toolu_1",
        name: "grep",
        arguments: '{"pattern":"foo"}',
      },
      {
        type: "function_call_output",
        call_id: "toolu_1",
        output: "bar",
      },
    ]);
  });

  test("converts anthropic tool schema to strict Responses tool schema", () => {
    const mapped = mapAnthropicToolsToResponsesTools([
      {
        name: "read_file",
        description: "Read a file",
        input_schema: {
          type: "object",
          properties: {
            path: { type: "string" },
            encoding: { type: "string", enum: ["utf8", "base64"] },
          },
          required: ["path"],
        },
      },
    ]);

    expect(mapped).toHaveLength(1);
    expect(mapped[0]).toMatchObject({
      type: "function",
      name: "read_file",
      strict: true,
    });
    expect(mapped[0].parameters).toMatchObject({
      type: "object",
      additionalProperties: false,
    });
  });

  test("adapts anthropic message request for Responses API", () => {
    const adapted = adaptAnthropicMessagesRequestForResponses(
      {
        model: "claude-sonnet-4-6",
        system: [{ type: "text", text: "Be concise." }],
        messages: [
          { role: "user", content: [{ type: "text", text: "Find foo" }] },
          {
            role: "assistant",
            content: [{ type: "tool_use", id: "toolu_1", name: "grep", input: { pattern: "foo" } }],
          },
        ],
        tools: [
          {
            name: "grep",
            description: "Search files",
            defer_loading: true,
            input_schema: {
              type: "object",
              properties: {
                pattern: { type: "string" },
              },
              required: ["pattern"],
            },
          },
          {
            name: "ToolSearch",
            description: "internal",
            input_schema: { type: "object", properties: {} },
          },
        ],
        tool_choice: { type: "any" },
        max_tokens: 1024,
        output_config: { effort: "high" },
        context_management: {
          edits: [{ type: "clear_tool_uses_20250919", trigger: "auto" }],
        },
      },
      {
        forcedModel: "gpt-5.3-codex",
        defaultReasoningEffort: "xhigh",
        preserveClientEffort: false,
        debug: false,
        safeMode: true,
        upstreamWireApi: "responses",
      }
    );

    expect(adapted.instructions).toBe("Be concise.");
    expect(adapted.input).toHaveLength(2);
    expect(adapted.tools).toHaveLength(1);
    expect(adapted.tools[0]).toMatchObject({ name: "grep", type: "function", strict: true });
    expect(adapted.tool_choice).toBe("required");
    expect(adapted.reasoning).toMatchObject({ effort: "high" });
    expect(adapted.messages).toBeUndefined();
    expect(adapted.system).toBeUndefined();
    expect(adapted.max_tokens).toBeUndefined();
    expect(adapted.context_management).toBeUndefined();
    expect(adapted.store).toBe(false);
  });

  test("maps Responses function calls back to anthropic tool_use blocks", () => {
    const mapped = mapResponsesOutputToAnthropicContent([
      {
        type: "message",
        content: [{ type: "output_text", text: "Checking files" }],
      },
      {
        type: "function_call",
        call_id: "toolu_2",
        name: "read_file",
        arguments: '{"path":"README.md"}',
      },
    ]);

    expect(mapped.stopReason).toBe("tool_use");
    expect(mapped.content).toEqual([
      { type: "text", text: "Checking files" },
      { type: "tool_use", id: "toolu_2", name: "read_file", input: { path: "README.md" } },
    ]);
  });

  test("rewrites messages path for responses upstreams", () => {
    expect(
      rewriteRequestPath(new URL("https://chatgpt.com/backend-api/codex"), "/v1/messages?x=1", "responses")
    ).toBe("/responses?x=1");
    expect(rewriteRequestPath(new URL("https://example.com/v1"), "/v1/messages", "responses")).toBe(
      "/v1/responses"
    );
    expect(rewriteRequestPath(new URL("https://example.com/v1"), "/v1/messages", "messages")).toBe(
      "/v1/messages"
    );
  });

  test("sanitizes unsupported anthropic tool fields", () => {
    const body: Record<string, any> = {
      tools: [{ name: "a", defer_loading: true }, { name: "b" }],
    };
    const removed = sanitizeToolFields(body);
    expect(removed).toBe(1);
    expect(body.tools[0].defer_loading).toBeUndefined();
  });
});
