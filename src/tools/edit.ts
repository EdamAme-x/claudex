/**
 * Edit tool - 文字列置換によるファイル編集
 */
import { readFile, writeFile } from "node:fs/promises";
import type { ToolResult } from "./index.js";

export async function executeEdit(
  filePath: string,
  oldString: string,
  newString: string,
): Promise<ToolResult> {
  const content = await readFile(filePath, "utf-8");

  // 一意性チェック
  const occurrences = content.split(oldString).length - 1;
  if (occurrences === 0) {
    return {
      output: `old_string not found in ${filePath}. Make sure it matches exactly.`,
      error: true,
    };
  }
  if (occurrences > 1) {
    return {
      output: `old_string found ${occurrences} times in ${filePath}. It must be unique. Add more surrounding context.`,
      error: true,
    };
  }

  const updated = content.replace(oldString, newString);
  await writeFile(filePath, updated, "utf-8");

  return { output: `Edited ${filePath}` };
}
