#!/usr/bin/env node

import http from "node:http";
import { Readable } from "node:stream";

const listenHost = process.env.CLAUDEX_LISTEN_HOST || "127.0.0.1";
const listenPort = Number(process.env.CLAUDEX_LISTEN_PORT || "18777");
const upstreamBaseUrl = process.env.CLAUDEX_UPSTREAM_BASE_URL;
const upstreamApiKey = process.env.CLAUDEX_UPSTREAM_API_KEY;
const forcedModel = process.env.CLAUDEX_FORCE_MODEL || "gpt-5.3-codex";
const defaultReasoningEffort = process.env.CLAUDEX_DEFAULT_REASONING_EFFORT || "xhigh";
const preserveClientEffort = process.env.CLAUDEX_PRESERVE_CLIENT_EFFORT === "1";

if (!upstreamBaseUrl || !upstreamApiKey) {
  console.error("claudex-proxy: missing CLAUDEX_UPSTREAM_BASE_URL or CLAUDEX_UPSTREAM_API_KEY");
  process.exit(1);
}

const upstream = new URL(upstreamBaseUrl);

function writeJson(res, statusCode, data) {
  const body = JSON.stringify(data);
  res.writeHead(statusCode, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function copyHeadersFromUpstream(upstreamHeaders) {
  const out = {};
  for (const [k, v] of upstreamHeaders) {
    if (k.toLowerCase() === "transfer-encoding") continue;
    out[k] = v;
  }
  return out;
}

function buildUpstreamUrl(reqUrl) {
  return new URL(reqUrl, upstream);
}

function approxTokenCount(obj) {
  const lines = [];
  if (Array.isArray(obj?.messages)) {
    for (const msg of obj.messages) {
      if (typeof msg?.content === "string") {
        lines.push(msg.content);
        continue;
      }
      if (Array.isArray(msg?.content)) {
        for (const part of msg.content) {
          if (typeof part?.text === "string") lines.push(part.text);
          if (typeof part?.content === "string") lines.push(part.content);
        }
      }
    }
  }
  const text = lines.join("\n");
  return Math.max(1, Math.ceil(text.length / 4));
}

function hasExplicitEffort(body) {
  if (typeof body?.effort === "string" && body.effort.length > 0) return true;
  if (typeof body?.output_config?.effort === "string" && body.output_config.effort.length > 0) return true;
  if (typeof body?.reasoning?.effort === "string" && body.reasoning.effort.length > 0) return true;
  return false;
}

function applyDefaultEffort(body) {
  // For gpt-5.3-codex, force xhigh by default.
  // If claudex was started with --effort, preserve user-specified effort.
  if (forcedModel !== "gpt-5.3-codex") return;
  if (preserveClientEffort && hasExplicitEffort(body)) return;

  if (typeof body.output_config !== "object" || body.output_config === null) {
    body.output_config = {};
  }
  body.output_config.effort = defaultReasoningEffort;

  if (typeof body.reasoning !== "object" || body.reasoning === null) {
    body.reasoning = {};
  }
  body.reasoning.effort = defaultReasoningEffort;
}

function sanitizeToolFields(body) {
  let removed = 0;
  if (!Array.isArray(body?.tools)) return removed;

  for (const tool of body.tools) {
    if (!tool || typeof tool !== "object") continue;

    // Some OpenAI-compatible Anthropic bridges reject this field.
    if ("defer_loading" in tool) {
      delete tool.defer_loading;
      removed += 1;
    }
  }
  return removed;
}

async function proxyRaw(req, res, reqBodyBuffer, reqUrl, overrideBodyJson = null) {
  const headers = {
    authorization: `Bearer ${upstreamApiKey}`,
  };

  // Preserve Anthropic-specific headers expected by the client/upstream bridge.
  if (req.headers["anthropic-version"]) {
    headers["anthropic-version"] = String(req.headers["anthropic-version"]);
  }
  if (req.headers["anthropic-beta"]) {
    headers["anthropic-beta"] = String(req.headers["anthropic-beta"]);
  }
  if (req.headers["content-type"]) {
    headers["content-type"] = String(req.headers["content-type"]);
  }

  let body = reqBodyBuffer;
  if (overrideBodyJson !== null) {
    const bodyText = JSON.stringify(overrideBodyJson);
    body = Buffer.from(bodyText);
    headers["content-type"] = "application/json";
    headers["content-length"] = String(body.length);
  } else if (body && body.length > 0) {
    headers["content-length"] = String(body.length);
  }

  const upstreamUrl = buildUpstreamUrl(reqUrl);
  const upstreamRes = await fetch(upstreamUrl, {
    method: req.method,
    headers,
    body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
  });

  res.writeHead(upstreamRes.status, copyHeadersFromUpstream(upstreamRes.headers));
  if (!upstreamRes.body) {
    res.end();
    return;
  }
  Readable.fromWeb(upstreamRes.body).pipe(res);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${listenHost}:${listenPort}`);
    const path = url.pathname;
    const method = req.method || "GET";

    if (method === "GET" && path === "/health") {
      writeJson(res, 200, {
        ok: true,
        forced_model: forcedModel,
        upstream: upstream.origin + upstream.pathname,
      });
      return;
    }

    // Claude Code may validate model access with GET /v1/models/{id}.
    // Some OpenAI-compatible gateways don't implement this route, so emulate success.
    if (method === "GET" && path.startsWith("/v1/models/")) {
      const requestedModel = decodeURIComponent(path.slice("/v1/models/".length));
      writeJson(res, 200, {
        id: requestedModel,
        object: "model",
        created: Math.floor(Date.now() / 1000),
        owned_by: "claudex",
      });
      return;
    }

    if (method === "POST" && path === "/v1/messages/count_tokens") {
      const bodyBuffer = await readBody(req);
      let parsed = {};
      try {
        parsed = bodyBuffer.length > 0 ? JSON.parse(bodyBuffer.toString("utf8")) : {};
      } catch {
        parsed = {};
      }
      writeJson(res, 200, { input_tokens: approxTokenCount(parsed) });
      return;
    }

    if (method === "POST" && path === "/v1/messages") {
      const bodyBuffer = await readBody(req);
      let parsed;
      try {
        parsed = JSON.parse(bodyBuffer.toString("utf8"));
      } catch {
        writeJson(res, 400, {
          type: "error",
          error: { type: "invalid_request_error", message: "Invalid JSON body" },
        });
        return;
      }

      const originalModel = parsed.model;
      parsed.model = forcedModel;
      applyDefaultEffort(parsed);
      const removedToolFields = sanitizeToolFields(parsed);
      if (process.env.CLAUDEX_DEBUG === "1") {
        console.error(
          `claudex-proxy model remap: ${String(originalModel)} -> ${forcedModel}, effort=${parsed.output_config?.effort ?? parsed.reasoning?.effort ?? "unset"}, preserve_client_effort=${preserveClientEffort}, removed_tool_fields=${removedToolFields}`
        );
      }
      await proxyRaw(req, res, bodyBuffer, url.pathname + url.search, parsed);
      return;
    }

    const bodyBuffer = await readBody(req);
    await proxyRaw(req, res, bodyBuffer, url.pathname + url.search);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    writeJson(res, 500, {
      type: "error",
      error: { type: "api_error", message: `claudex-proxy internal error: ${msg}` },
    });
  }
});

server.listen(listenPort, listenHost, () => {
  console.error(`claudex-proxy listening on http://${listenHost}:${listenPort}`);
});
