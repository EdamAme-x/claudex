/**
 * StatusBar - 画面下部のステータスバー
 */
import React from "react";
import { Box, Text } from "ink";

interface Props {
  tokens: number;
  model: string;
  autoApprove: boolean;
  width: number;
}

export function StatusBar({ tokens, model, autoApprove, width }: Props) {
  const cwd = process.cwd();
  const dir =
    cwd.length > 30 ? "..." + cwd.slice(cwd.length - 27) : cwd;

  return (
    <Box justifyContent="space-between" paddingX={1}>
      <Text dimColor>
        {dir}
      </Text>
      <Text dimColor>
        {autoApprove ? (
          <Text color="yellow">AUTO </Text>
        ) : null}
        {tokens > 0 ? `${(tokens / 1000).toFixed(1)}k tokens | ` : ""}
        {model} | <Text color="cyan">Ctrl+C</Text> exit
      </Text>
    </Box>
  );
}
