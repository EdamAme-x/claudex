/**
 * OpenAI OAuth PKCE login flow
 * ChatGPTアカウントでログイン（APIキー不要）
 */
import { createServer } from "node:http";
import { createHash, randomBytes } from "node:crypto";
import { URL } from "node:url";
import open from "open";
import { loadAuth, saveAuth, type AuthData } from "./storage.js";

const ISSUER = "https://auth.openai.com";
const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const SCOPES =
  "openid profile email offline_access api.connectors.read api.connectors.invoke";
const CALLBACK_PORT = 1477; // 1455はCodex CLIが使用するため別ポート
const REDIRECT_URI = `http://localhost:${CALLBACK_PORT}/auth/callback`;

interface PkceCodes {
  verifier: string;
  challenge: string;
}

function generatePkce(): PkceCodes {
  const verifier = randomBytes(64)
    .toString("base64url")
    .replace(/[^a-zA-Z0-9\-._~]/g, "");
  const challenge = createHash("sha256").update(verifier).digest("base64url");
  return { verifier, challenge };
}

function buildAuthorizeUrl(pkce: PkceCodes, state: string): string {
  const params = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    scope: SCOPES,
    code_challenge: pkce.challenge,
    code_challenge_method: "S256",
    state,
    codex_cli_simplified_flow: "true",
  });
  return `${ISSUER}/oauth/authorize?${params}`;
}

async function exchangeCodeForTokens(
  code: string,
  pkce: PkceCodes
): Promise<{ id_token: string; access_token: string; refresh_token: string }> {
  const body = new URLSearchParams({
    grant_type: "authorization_code",
    code,
    redirect_uri: REDIRECT_URI,
    client_id: CLIENT_ID,
    code_verifier: pkce.verifier,
  });

  const resp = await fetch(`${ISSUER}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Token exchange failed (${resp.status}): ${text}`);
  }

  return resp.json();
}

async function exchangeForApiKey(idToken: string): Promise<string> {
  const body = new URLSearchParams({
    grant_type: "urn:ietf:params:oauth:grant-type:token-exchange",
    client_id: CLIENT_ID,
    requested_token: "openai-api-key",
    subject_token: idToken,
    subject_token_type: "urn:ietf:params:oauth:token-type:id_token",
  });

  const resp = await fetch(`${ISSUER}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API key exchange failed (${resp.status}): ${text}`);
  }

  const data = (await resp.json()) as { access_token: string };
  return data.access_token;
}

function parseIdToken(jwt: string): {
  email?: string;
  plan_type?: string;
  user_id?: string;
  account_id?: string;
} {
  try {
    const payload = JSON.parse(
      Buffer.from(jwt.split(".")[1], "base64url").toString()
    );
    const auth = payload["https://api.openai.com/auth"] || {};
    const profile = payload["https://api.openai.com/profile"] || {};
    return {
      email: profile.email || payload.email,
      plan_type: auth.chatgpt_plan_type,
      user_id: auth.chatgpt_user_id,
      account_id: auth.chatgpt_account_id,
    };
  } catch {
    return {};
  }
}

/**
 * ブラウザベースのOAuthログインフローを実行
 */
export async function login(): Promise<AuthData> {
  const pkce = generatePkce();
  const state = randomBytes(32).toString("base64url");

  return new Promise((resolve, reject) => {
    const server = createServer(async (req, res) => {
      const url = new URL(req.url || "/", `http://localhost:${CALLBACK_PORT}`);

      if (url.pathname === "/auth/callback") {
        const code = url.searchParams.get("code");
        const error = url.searchParams.get("error");
        const returnedState = url.searchParams.get("state");

        if (error) {
          res.writeHead(200, { "Content-Type": "text/html" });
          res.end(
            "<html><body><h1>Login failed</h1><p>You can close this tab.</p></body></html>"
          );
          server.close();
          reject(new Error(`OAuth error: ${error}`));
          return;
        }

        if (returnedState !== state) {
          res.writeHead(400, { "Content-Type": "text/html" });
          res.end("<html><body><h1>State mismatch</h1></body></html>");
          server.close();
          reject(new Error("OAuth state mismatch"));
          return;
        }

        if (!code) {
          res.writeHead(400, { "Content-Type": "text/html" });
          res.end("<html><body><h1>No code received</h1></body></html>");
          server.close();
          reject(new Error("No authorization code"));
          return;
        }

        try {
          // Step 1: Exchange code for tokens
          const tokens = await exchangeCodeForTokens(code, pkce);

          // Step 2: Parse ID token for user info
          const userInfo = parseIdToken(tokens.id_token);

          // Step 3: Exchange for API key
          const apiKey = await exchangeForApiKey(tokens.id_token);

          const authData: AuthData = {
            api_key: apiKey,
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token,
            id_token: tokens.id_token,
            email: userInfo.email,
            plan_type: userInfo.plan_type,
            account_id: userInfo.account_id,
            last_refresh: new Date().toISOString(),
          };

          await saveAuth(authData);

          res.writeHead(200, { "Content-Type": "text/html" });
          res.end(
            `<html><body style="font-family:system-ui;text-align:center;padding:60px"><h1>Logged in to claudex!</h1><p>${userInfo.email || ""}</p><p style="color:#666">You can close this tab.</p></body></html>`
          );
          server.close();
          resolve(authData);
        } catch (err) {
          res.writeHead(500, { "Content-Type": "text/html" });
          res.end("<html><body><h1>Token exchange failed</h1></body></html>");
          server.close();
          reject(err);
        }
      }
    });

    server.listen(CALLBACK_PORT, "127.0.0.1", () => {
      const authorizeUrl = buildAuthorizeUrl(pkce, state);
      open(authorizeUrl);
    });

    server.on("error", (err) => {
      reject(
        new Error(
          `Failed to start OAuth server on port ${CALLBACK_PORT}: ${err.message}`
        )
      );
    });

    // 5分でタイムアウト
    setTimeout(() => {
      server.close();
      reject(new Error("Login timed out (5 minutes)"));
    }, 5 * 60 * 1000);
  });
}

/**
 * トークンリフレッシュ
 */
export async function refreshTokens(auth: AuthData): Promise<AuthData> {
  const resp = await fetch(`${ISSUER}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id: CLIENT_ID,
      grant_type: "refresh_token",
      refresh_token: auth.refresh_token,
    }),
  });

  if (!resp.ok) {
    throw new Error(`Token refresh failed (${resp.status})`);
  }

  const data = (await resp.json()) as {
    id_token?: string;
    access_token?: string;
    refresh_token?: string;
  };

  const newIdToken = data.id_token || auth.id_token;
  const userInfo = parseIdToken(newIdToken);

  // Refresh後にAPI keyを再取得
  const apiKey = await exchangeForApiKey(newIdToken);

  const updated: AuthData = {
    ...auth,
    api_key: apiKey,
    access_token: data.access_token || auth.access_token,
    refresh_token: data.refresh_token || auth.refresh_token,
    id_token: newIdToken,
    email: userInfo.email || auth.email,
    plan_type: userInfo.plan_type || auth.plan_type,
    account_id: userInfo.account_id || auth.account_id,
    last_refresh: new Date().toISOString(),
  };

  await saveAuth(updated);
  return updated;
}

/**
 * 認証済みか確認し、必要ならリフレッシュ
 */
export async function ensureAuth(): Promise<AuthData> {
  const auth = await loadAuth();
  if (!auth) {
    throw new Error("Not logged in. Run: claudex login");
  }

  // 8日以上経過していたらリフレッシュ
  const lastRefresh = new Date(auth.last_refresh || 0);
  const daysSinceRefresh =
    (Date.now() - lastRefresh.getTime()) / (1000 * 60 * 60 * 24);

  if (daysSinceRefresh > 8) {
    try {
      return await refreshTokens(auth);
    } catch {
      // リフレッシュ失敗時は再ログインが必要
      throw new Error("Session expired. Run: claudex login");
    }
  }

  return auth;
}
