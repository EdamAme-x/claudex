/**
 * Read tool - ファイル読み取り（行番号付き）
 */
import { readFile } from "node:fs/promises";
import type { ToolResult } from "./index.js";

export async function executeRead(
  filePath: string,
  offset?: number,
  limit?: number,
): Promise<ToolResult> {
  const content = await readFile(filePath, "utf-8");
  const lines = content.split("\n");

  const start = (offset || 1) - 1;
  const end = limit ? start + limit : lines.length;
  const slice = lines.slice(start, end);

  const numbered = slice
    .map((line, i) => {
      const lineNum = String(start + i + 1).padStart(6, " ");
      // 長い行は切り詰め
      const truncated = line.length > 2000 ? line.slice(0, 2000) + "..." : line;
      return `${lineNum}\t${truncated}`;
    })
    .join("\n");

  return { output: numbered };
}
