/**
 * メインアプリケーション - Claude Code風ターミナルUI
 */
import React, { useState, useCallback, useEffect, useRef } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import { InputBox } from "./InputBox.js";
import { MessageView } from "./MessageView.js";
import { ToolStatus } from "./ToolStatus.js";
import { StatusBar } from "./StatusBar.js";
import { PermissionPrompt } from "./PermissionPrompt.js";
import { OpenAIClient, type Message, type ToolCall } from "../api/openai.js";
import {
  toolDefinitions,
  executeTool,
  TOOL_LABELS,
  type ToolResult,
} from "../tools/index.js";
import { buildSystemPrompt } from "../prompts/system.js";

interface ConversationEntry {
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  toolCallId?: string;
  isStreaming?: boolean;
  isError?: boolean;
}

interface PendingPermission {
  toolCall: ToolCall;
  resolve: (allowed: boolean) => void;
}

interface Props {
  client: OpenAIClient;
  email?: string;
  planType?: string;
}

// ツール実行に許可が必要なツール
const DANGEROUS_TOOLS = new Set(["Bash", "Write", "Edit"]);

export function App({ client, email, planType }: Props) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    { role: "system", content: buildSystemPrompt() },
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTool, setCurrentTool] = useState<string | null>(null);
  const [pendingPermission, setPendingPermission] =
    useState<PendingPermission | null>(null);
  const [tokenUsage, setTokenUsage] = useState({
    prompt: 0,
    completion: 0,
    total: 0,
  });
  const [autoApprove, setAutoApprove] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const termWidth = stdout?.columns || 80;

  const handleSubmit = useCallback(
    async (input: string) => {
      if (isProcessing) return;
      const trimmed = input.trim();
      if (!trimmed) return;

      // Special commands
      if (trimmed === "/exit" || trimmed === "/quit") {
        exit();
        return;
      }
      if (trimmed === "/clear") {
        setConversation([]);
        setMessages([{ role: "system", content: buildSystemPrompt() }]);
        return;
      }
      if (trimmed === "/auto") {
        setAutoApprove((v) => !v);
        setConversation((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Auto-approve: ${!autoApprove ? "ON" : "OFF"}`,
          },
        ]);
        return;
      }
      if (trimmed === "/model") {
        setConversation((prev) => [
          ...prev,
          { role: "assistant", content: `Current model: ${client.model}` },
        ]);
        return;
      }
      if (trimmed.startsWith("/model ")) {
        const newModel = trimmed.slice(7).trim();
        client.model = newModel;
        setConversation((prev) => [
          ...prev,
          { role: "assistant", content: `Model changed to: ${newModel}` },
        ]);
        return;
      }

      // ユーザーメッセージを追加
      const userEntry: ConversationEntry = { role: "user", content: trimmed };
      setConversation((prev) => [...prev, userEntry]);

      const userMsg: Message = { role: "user", content: trimmed };
      const updatedMessages = [...messages, userMsg];
      setMessages(updatedMessages);
      setIsProcessing(true);

      await runAssistant(updatedMessages);
    },
    [isProcessing, messages, autoApprove],
  );

  const runAssistant = async (currentMessages: Message[]) => {
    let assistantContent = "";
    const streamingIdx = conversation.length + 1; // +1 for the user message we just added

    // Streaming entry
    setConversation((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);

    try {
      const toolCalls: ToolCall[] = [];

      for await (const delta of client.stream(
        currentMessages,
        toolDefinitions,
      )) {
        if (delta.type === "content" && delta.content) {
          assistantContent += delta.content;
          setConversation((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: assistantContent,
              isStreaming: true,
            };
            return updated;
          });
        }

        if (delta.type === "tool_call" && delta.tool_call) {
          toolCalls.push(delta.tool_call);
        }

        if (delta.type === "done" && delta.usage) {
          setTokenUsage((prev) => ({
            prompt: prev.prompt + (delta.usage?.prompt_tokens || 0),
            completion: prev.completion + (delta.usage?.completion_tokens || 0),
            total: prev.total + (delta.usage?.total_tokens || 0),
          }));
        }

        if (delta.type === "error") {
          setConversation((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "assistant",
              content: delta.error || "Unknown error",
              isError: true,
            };
            return updated;
          });
          setIsProcessing(false);
          return;
        }
      }

      // Stop streaming indicator
      setConversation((prev) => {
        const updated = [...prev];
        if (updated[updated.length - 1]?.isStreaming) {
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            isStreaming: false,
          };
        }
        return updated;
      });

      // Assistantメッセージを記録
      const assistantMsg: Message = {
        role: "assistant",
        content: assistantContent || "",
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
      };
      const nextMessages = [...currentMessages, assistantMsg];

      // ツールコールがあれば実行
      if (toolCalls.length > 0) {
        const toolMessages: Message[] = [];

        for (const tc of toolCalls) {
          const toolName = tc.function.name;
          let args: Record<string, unknown>;
          try {
            args = JSON.parse(tc.function.arguments);
          } catch {
            args = {};
          }

          // 許可確認
          if (DANGEROUS_TOOLS.has(toolName) && !autoApprove) {
            const allowed = await requestPermission(tc);
            if (!allowed) {
              const deniedResult: ToolResult = {
                output: "User denied this tool execution.",
                error: true,
              };
              setConversation((prev) => [
                ...prev,
                {
                  role: "tool",
                  content: deniedResult.output,
                  toolName,
                  isError: true,
                },
              ]);
              toolMessages.push({
                role: "tool",
                content: deniedResult.output,
                tool_call_id: tc.id,
              });
              continue;
            }
          }

          // ツール実行
          setCurrentTool(TOOL_LABELS[toolName] || toolName);

          const argSummary =
            toolName === "Bash"
              ? (args.command as string)
              : toolName === "Read" || toolName === "Write" || toolName === "Edit"
                ? (args.file_path as string)
                : toolName === "Glob"
                  ? (args.pattern as string)
                  : (args.pattern as string);

          setConversation((prev) => [
            ...prev,
            {
              role: "tool",
              content: `${TOOL_LABELS[toolName] || toolName}: ${argSummary || ""}`,
              toolName,
              isStreaming: true,
            },
          ]);

          const result = await executeTool(tc);
          setCurrentTool(null);

          setConversation((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              role: "tool",
              content: result.output,
              toolName,
              isError: result.error,
              isStreaming: false,
            };
            return updated;
          });

          toolMessages.push({
            role: "tool",
            content: result.output,
            tool_call_id: tc.id,
          });
        }

        // ツール結果をメッセージに追加して再度呼び出し
        const messagesWithTools = [...nextMessages, ...toolMessages];
        setMessages(messagesWithTools);
        await runAssistant(messagesWithTools);
      } else {
        setMessages(nextMessages);
        setIsProcessing(false);
      }
    } catch (err: any) {
      setConversation((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.isStreaming) {
          updated[updated.length - 1] = {
            role: "assistant",
            content: `Error: ${err.message}`,
            isError: true,
            isStreaming: false,
          };
        } else {
          updated.push({
            role: "assistant",
            content: `Error: ${err.message}`,
            isError: true,
          });
        }
        return updated;
      });
      setIsProcessing(false);
    }
  };

  const requestPermission = (toolCall: ToolCall): Promise<boolean> => {
    return new Promise((resolve) => {
      setPendingPermission({ toolCall, resolve });
    });
  };

  const handlePermissionResponse = useCallback(
    (allowed: boolean) => {
      if (pendingPermission) {
        pendingPermission.resolve(allowed);
        setPendingPermission(null);
      }
    },
    [pendingPermission],
  );

  // Ctrl+C でキャンセル
  useInput((input, key) => {
    if (key.ctrl && input === "c") {
      if (isProcessing) {
        abortRef.current?.abort();
        setIsProcessing(false);
        setCurrentTool(null);
      } else {
        exit();
      }
    }
  });

  return (
    <Box flexDirection="column" width={termWidth}>
      {/* Header */}
      <Box
        borderStyle="round"
        borderColor="cyan"
        paddingX={1}
        justifyContent="space-between"
      >
        <Text bold color="cyan">
          claudex
        </Text>
        <Text dimColor>
          {client.model} | {email || "logged in"}
          {planType ? ` (${planType})` : ""}
        </Text>
      </Box>

      {/* Conversation */}
      <Box flexDirection="column" paddingX={1} flexGrow={1}>
        {conversation.map((entry, i) => (
          <MessageView key={i} entry={entry} width={termWidth - 4} />
        ))}
      </Box>

      {/* Tool status */}
      {currentTool && <ToolStatus label={currentTool} />}

      {/* Permission prompt */}
      {pendingPermission && (
        <PermissionPrompt
          toolCall={pendingPermission.toolCall}
          onResponse={handlePermissionResponse}
        />
      )}

      {/* Input */}
      {!pendingPermission && (
        <InputBox
          onSubmit={handleSubmit}
          isProcessing={isProcessing}
          placeholder={isProcessing ? "Processing..." : "Message claudex..."}
        />
      )}

      {/* Status bar */}
      <StatusBar
        tokens={tokenUsage.total}
        model={client.model}
        autoApprove={autoApprove}
        width={termWidth}
      />
    </Box>
  );
}
