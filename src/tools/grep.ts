/**
 * Grep tool - ファイル内容検索
 */
import { spawn } from "node:child_process";
import type { ToolResult } from "./index.js";

export async function executeGrep(
  pattern: string,
  searchPath?: string,
  include?: string,
): Promise<ToolResult> {
  const cwd = searchPath || process.cwd();
  const args = [
    "-rn",
    "--color=never",
    "--max-count=100",
    "-E",
    pattern,
  ];

  if (include) {
    args.push(`--include=${include}`);
  }

  // node_modules と .git を除外
  args.push("--exclude-dir=node_modules", "--exclude-dir=.git");
  args.push(cwd);

  return new Promise((resolve) => {
    const proc = spawn("grep", args, { timeout: 30000 });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      if (code === 1 && !stdout) {
        resolve({ output: "No matches found." });
      } else if (code !== 0 && code !== 1) {
        resolve({ output: stderr || `grep exited with code ${code}`, error: true });
      } else {
        // 結果が大きすぎる場合は切り詰め
        const maxLen = 50000;
        const trimmed =
          stdout.length > maxLen
            ? stdout.slice(0, maxLen) + "\n... (truncated)"
            : stdout;
        resolve({ output: trimmed || "No matches found." });
      }
    });

    proc.on("error", () => {
      // grep が無い場合はNode.jsで代替
      resolve({ output: "grep command not available", error: true });
    });
  });
}
