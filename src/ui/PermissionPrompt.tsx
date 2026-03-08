/**
 * PermissionPrompt - ツール実行許可ダイアログ（Claude Code風）
 */
import React from "react";
import { Box, Text, useInput } from "ink";
import type { ToolCall } from "../api/openai.js";

interface Props {
  toolCall: ToolCall;
  onResponse: (allowed: boolean) => void;
}

export function PermissionPrompt({ toolCall, onResponse }: Props) {
  const name = toolCall.function.name;
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(toolCall.function.arguments);
  } catch {
    args = {};
  }

  // ツールの詳細表示
  let detail = "";
  switch (name) {
    case "Bash":
      detail = String(args.command || "");
      break;
    case "Write":
      detail = `${args.file_path} (${String(args.content || "").split("\n").length} lines)`;
      break;
    case "Edit":
      detail = String(args.file_path || "");
      break;
    default:
      detail = JSON.stringify(args).slice(0, 200);
  }

  useInput((input, key) => {
    if (input === "y" || input === "Y" || key.return) {
      onResponse(true);
    } else if (input === "n" || input === "N" || key.escape) {
      onResponse(false);
    }
  });

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="yellow"
      paddingX={1}
      marginY={0}
    >
      <Box>
        <Text bold color="yellow">
          Allow {name}?
        </Text>
      </Box>
      <Box paddingLeft={1} marginY={0}>
        <Text wrap="wrap">{detail}</Text>
      </Box>
      <Box marginTop={0}>
        <Text dimColor>
          [<Text color="green" bold>Y</Text>]es{" "}
          [<Text color="red" bold>N</Text>]o{" "}
          <Text dimColor>(use /auto to auto-approve all)</Text>
        </Text>
      </Box>
    </Box>
  );
}
