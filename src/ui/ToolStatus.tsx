/**
 * ToolStatus - ツール実行中のステータス表示
 */
import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";

interface Props {
  label: string;
}

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function ToolStatus({ label }: Props) {
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setFrame((f) => (f + 1) % SPINNER_FRAMES.length);
    }, 80);
    return () => clearInterval(timer);
  }, []);

  return (
    <Box paddingX={1}>
      <Text color="yellow">
        {SPINNER_FRAMES[frame]} {label}...
      </Text>
    </Box>
  );
}
