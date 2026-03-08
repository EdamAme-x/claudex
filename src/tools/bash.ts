/**
 * Bash tool - コマンド実行
 */
import { spawn } from "node:child_process";
import type { ToolResult } from "./index.js";

export async function executeBash(
  command: string,
  timeout = 120000,
): Promise<ToolResult> {
  return new Promise((resolve) => {
    const proc = spawn("bash", ["-c", command], {
      cwd: process.cwd(),
      env: { ...process.env },
      timeout,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });
    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("close", (code) => {
      const output = stdout + (stderr ? `\n${stderr}` : "");
      // 出力が大きすぎる場合は切り詰め
      const maxLen = 100000;
      const trimmed =
        output.length > maxLen
          ? output.slice(0, maxLen) + `\n... (truncated, ${output.length} total chars)`
          : output;

      if (code !== 0) {
        resolve({ output: trimmed || `Exit code: ${code}`, error: true });
      } else {
        resolve({ output: trimmed || "(no output)" });
      }
    });

    proc.on("error", (err) => {
      resolve({ output: err.message, error: true });
    });
  });
}
