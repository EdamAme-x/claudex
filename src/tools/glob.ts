/**
 * Glob tool - ファイルパターンマッチング
 */
import fastGlob from "fast-glob";
import type { ToolResult } from "./index.js";

export async function executeGlob(
  pattern: string,
  searchPath?: string,
): Promise<ToolResult> {
  const cwd = searchPath || process.cwd();
  const files = await fastGlob(pattern, {
    cwd,
    absolute: true,
    dot: false,
    ignore: ["**/node_modules/**", "**/.git/**"],
    onlyFiles: true,
  });

  if (files.length === 0) {
    return { output: "No files matched the pattern." };
  }

  // 最大500件
  const limited = files.slice(0, 500);
  const result = limited.join("\n");
  const suffix =
    files.length > 500 ? `\n... and ${files.length - 500} more files` : "";

  return { output: result + suffix };
}
