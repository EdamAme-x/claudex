/**
 * claudex設定管理
 * ~/.claudex/config.json に永続化
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { homedir } from "node:os";

export interface ClaudexConfig {
  theme: "dark" | "light" | "auto";
  permMode: "normal" | "auto-approve" | "plan-mode";
  model: string;
  onboarded: boolean;
  version: string;
}

const DEFAULTS: ClaudexConfig = {
  theme: "dark",
  permMode: "normal",
  model: "o4-mini",
  onboarded: false,
  version: "0.1.0",
};

function getConfigDir(): string {
  return process.env.CLAUDEX_HOME || join(homedir(), ".claudex");
}

function getConfigFile(): string {
  return join(getConfigDir(), "config.json");
}

export async function loadConfig(): Promise<ClaudexConfig> {
  try {
    const data = await readFile(getConfigFile(), "utf-8");
    return { ...DEFAULTS, ...JSON.parse(data) };
  } catch {
    return { ...DEFAULTS };
  }
}

export async function saveConfig(config: ClaudexConfig): Promise<void> {
  const dir = getConfigDir();
  await mkdir(dir, { recursive: true });
  await writeFile(getConfigFile(), JSON.stringify(config, null, 2), "utf-8");
}

export async function isFirstRun(): Promise<boolean> {
  const config = await loadConfig();
  return !config.onboarded;
}
