/**
 * OpenAI API クライアント（ストリーミング対応）
 */

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface StreamDelta {
  type: "content" | "tool_call" | "done" | "error";
  content?: string;
  tool_call?: ToolCall;
  error?: string;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

const DEFAULT_BASE_URL = "https://api.openai.com/v1";

export class OpenAIClient {
  private apiKey: string;
  private baseUrl: string;
  public model: string;

  constructor(apiKey: string, model = "o4-mini", baseUrl?: string) {
    this.apiKey = apiKey;
    this.model = model;
    this.baseUrl = baseUrl || DEFAULT_BASE_URL;
  }

  /**
   * ストリーミングChat Completions
   */
  async *stream(
    messages: Message[],
    tools?: ToolDefinition[],
  ): AsyncGenerator<StreamDelta> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages,
      stream: true,
      stream_options: { include_usage: true },
    };

    if (tools && tools.length > 0) {
      body.tools = tools;
      body.tool_choice = "auto";
    }

    const resp = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text();
      yield { type: "error", error: `API error (${resp.status}): ${text}` };
      return;
    }

    const reader = resp.body?.getReader();
    if (!reader) {
      yield { type: "error", error: "No response body" };
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    // Track partial tool calls being built up
    const partialToolCalls = new Map<number, { id: string; name: string; args: string }>();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") {
          // Flush any remaining tool calls
          for (const [, tc] of partialToolCalls) {
            yield {
              type: "tool_call",
              tool_call: {
                id: tc.id,
                type: "function",
                function: { name: tc.name, arguments: tc.args },
              },
            };
          }
          partialToolCalls.clear();
          yield { type: "done" };
          return;
        }

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta;
          const usage = parsed.usage;

          if (usage) {
            yield { type: "done", usage };
            return;
          }

          if (delta?.content) {
            yield { type: "content", content: delta.content };
          }

          if (delta?.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index ?? 0;
              if (tc.id) {
                // New tool call starting
                partialToolCalls.set(idx, {
                  id: tc.id,
                  name: tc.function?.name || "",
                  args: tc.function?.arguments || "",
                });
              } else {
                // Continuation of existing tool call
                const existing = partialToolCalls.get(idx);
                if (existing) {
                  if (tc.function?.name) existing.name += tc.function.name;
                  if (tc.function?.arguments) existing.args += tc.function.arguments;
                }
              }
            }
          }

          // If finish_reason is "tool_calls", flush
          if (parsed.choices?.[0]?.finish_reason === "tool_calls") {
            for (const [, tc] of partialToolCalls) {
              yield {
                type: "tool_call",
                tool_call: {
                  id: tc.id,
                  type: "function",
                  function: { name: tc.name, arguments: tc.args },
                },
              };
            }
            partialToolCalls.clear();
          }
        } catch {
          // Skip unparseable lines
        }
      }
    }
  }

  /**
   * 非ストリーミング呼び出し
   */
  async complete(
    messages: Message[],
    tools?: ToolDefinition[],
  ): Promise<{ message: Message; usage?: { prompt_tokens: number; completion_tokens: number } }> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages,
    };

    if (tools && tools.length > 0) {
      body.tools = tools;
      body.tool_choice = "auto";
    }

    const resp = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`API error (${resp.status}): ${text}`);
    }

    const data = await resp.json() as any;
    return {
      message: data.choices[0].message,
      usage: data.usage,
    };
  }
}
