/**
 * MessageView - メッセージ表示コンポーネント
 */
import React from "react";
import { Box, Text } from "ink";

interface ConversationEntry {
  role: "user" | "assistant" | "tool";
  content: string;
  toolName?: string;
  toolCallId?: string;
  isStreaming?: boolean;
  isError?: boolean;
}

interface Props {
  entry: ConversationEntry;
  width: number;
}

export function MessageView({ entry, width }: Props) {
  const { role, content, toolName, isStreaming, isError } = entry;

  if (role === "user") {
    return (
      <Box marginY={0} flexDirection="column">
        <Box>
          <Text bold color="blue">
            You
          </Text>
        </Box>
        <Box paddingLeft={1}>
          <Text wrap="wrap">{content}</Text>
        </Box>
        <Text dimColor>{"─".repeat(Math.min(width, 60))}</Text>
      </Box>
    );
  }

  if (role === "tool") {
    // ツール実行結果
    const icon = isError ? "✗" : "✓";
    const color = isError ? "red" : "green";
    const label = toolName || "Tool";

    if (isStreaming) {
      return (
        <Box marginY={0} paddingLeft={1}>
          <Text color="yellow">⟳ {content}</Text>
        </Box>
      );
    }

    // ツール結果は折りたたんで表示（長い場合）
    const lines = content.split("\n");
    const maxLines = 20;
    const truncated = lines.length > maxLines;
    const displayContent = truncated
      ? lines.slice(0, maxLines).join("\n") +
        `\n... (${lines.length - maxLines} more lines)`
      : content;

    return (
      <Box marginY={0} flexDirection="column" paddingLeft={1}>
        <Text color={color}>
          {icon} {label}
        </Text>
        <Box
          borderStyle="single"
          borderColor="gray"
          paddingX={1}
          marginLeft={1}
        >
          <Text dimColor wrap="wrap">
            {displayContent}
          </Text>
        </Box>
      </Box>
    );
  }

  // Assistant
  if (isError) {
    return (
      <Box marginY={0} flexDirection="column">
        <Text bold color="red">
          Error
        </Text>
        <Box paddingLeft={1}>
          <Text color="red" wrap="wrap">{content}</Text>
        </Box>
      </Box>
    );
  }

  return (
    <Box marginY={0} flexDirection="column">
      <Box>
        <Text bold color="magenta">
          claudex
        </Text>
        {isStreaming && <Text color="yellow"> ●</Text>}
      </Box>
      <Box paddingLeft={1} flexDirection="column">
        <Text wrap="wrap">{content}</Text>
      </Box>
    </Box>
  );
}
