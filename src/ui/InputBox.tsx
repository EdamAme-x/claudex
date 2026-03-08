/**
 * InputBox - Claude Code風の入力コンポーネント
 */
import React, { useState, useCallback } from "react";
import { Box, Text, useInput } from "ink";

interface Props {
  onSubmit: (text: string) => void;
  isProcessing: boolean;
  placeholder: string;
}

export function InputBox({ onSubmit, isProcessing, placeholder }: Props) {
  const [value, setValue] = useState("");
  const [cursorPos, setCursorPos] = useState(0);

  useInput(
    (input, key) => {
      if (isProcessing) return;

      if (key.return) {
        if (value.trim()) {
          onSubmit(value);
          setValue("");
          setCursorPos(0);
        }
        return;
      }

      if (key.backspace || key.delete) {
        if (cursorPos > 0) {
          const newVal = value.slice(0, cursorPos - 1) + value.slice(cursorPos);
          setValue(newVal);
          setCursorPos(cursorPos - 1);
        }
        return;
      }

      if (key.leftArrow) {
        setCursorPos(Math.max(0, cursorPos - 1));
        return;
      }
      if (key.rightArrow) {
        setCursorPos(Math.min(value.length, cursorPos + 1));
        return;
      }

      // Ctrl+A: 先頭
      if (key.ctrl && input === "a") {
        setCursorPos(0);
        return;
      }
      // Ctrl+E: 末尾
      if (key.ctrl && input === "e") {
        setCursorPos(value.length);
        return;
      }
      // Ctrl+U: 行クリア
      if (key.ctrl && input === "u") {
        setValue("");
        setCursorPos(0);
        return;
      }
      // Ctrl+W: 単語削除
      if (key.ctrl && input === "w") {
        const before = value.slice(0, cursorPos);
        const after = value.slice(cursorPos);
        const trimmed = before.replace(/\S+\s*$/, "");
        setValue(trimmed + after);
        setCursorPos(trimmed.length);
        return;
      }

      if (input && !key.ctrl && !key.meta) {
        const newVal =
          value.slice(0, cursorPos) + input + value.slice(cursorPos);
        setValue(newVal);
        setCursorPos(cursorPos + input.length);
      }
    },
    { isActive: !isProcessing },
  );

  // カーソル表示
  const before = value.slice(0, cursorPos);
  const cursor = value[cursorPos] || " ";
  const after = value.slice(cursorPos + 1);

  return (
    <Box borderStyle="round" borderColor={isProcessing ? "gray" : "green"} paddingX={1}>
      <Text color="green" bold>
        {">"}{" "}
      </Text>
      {value.length === 0 && !isProcessing ? (
        <Text dimColor>{placeholder}</Text>
      ) : (
        <Text>
          {before}
          <Text inverse>{cursor}</Text>
          {after}
        </Text>
      )}
      {isProcessing && (
        <Text dimColor> {placeholder}</Text>
      )}
    </Box>
  );
}
