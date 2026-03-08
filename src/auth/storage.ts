/**
 * 認証データのファイルベース永続化
 */
import { readFile, writeFile, mkdir, chmod } from "node:fs/promises";
import { join } from "node:path";
import { homedir } from "node:os";

export interface AuthData {
  api_key: string;
  access_token: string;
  refresh_token: string;
  id_token: string;
  email?: string;
  plan_type?: string;
  account_id?: string;
  last_refresh: string;
}

function getAuthDir(): string {
  // CLAUDEX_HOME環境変数でオーバーライド可能（~/.claude/ や ~/.codex/ と競合しない）
  return process.env.CLAUDEX_HOME || join(homedir(), ".claudex");
}

function getAuthFile(): string {
  return join(getAuthDir(), "auth.json");
}

export async function loadAuth(): Promise<AuthData | null> {
  try {
    const data = await readFile(getAuthFile(), "utf-8");
    return JSON.parse(data) as AuthData;
  } catch {
    return null;
  }
}

export async function saveAuth(auth: AuthData): Promise<void> {
  const dir = getAuthDir();
  await mkdir(dir, { recursive: true });
  const file = getAuthFile();
  await writeFile(file, JSON.stringify(auth, null, 2), { mode: 0o600 });
  try {
    await chmod(file, 0o600);
  } catch {
    // Windows doesn't support chmod the same way
  }
}

export async function clearAuth(): Promise<void> {
  try {
    const { unlink } = await import("node:fs/promises");
    await unlink(getAuthFile());
  } catch {
    // Already gone
  }
}
