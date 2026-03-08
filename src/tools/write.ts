/**
 * Write tool - ファイル書き込み
 */
import { writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type { ToolResult } from "./index.js";

export async function executeWrite(
  filePath: string,
  content: string,
): Promise<ToolResult> {
  await mkdir(dirname(filePath), { recursive: true });
  await writeFile(filePath, content, "utf-8");
  const lines = content.split("\n").length;
  return { output: `Wrote ${lines} lines to ${filePath}` };
}
