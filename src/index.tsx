#!/usr/bin/env node
/**
 * claudex - Claude Code UI + OpenAI models
 * ChatGPTアカウントでログイン、APIキー不要
 */
import React, { useState, useEffect } from "react";
import { render, Box, Text } from "ink";
import { App } from "./ui/App.js";
import { Onboarding } from "./ui/Onboarding.js";
import { login, ensureAuth } from "./auth/oauth.js";
import { loadAuth, clearAuth } from "./auth/storage.js";
import { loadConfig, saveConfig, isFirstRun, type ClaudexConfig } from "./config.js";
import { OpenAIClient } from "./api/openai.js";

const VERSION = "0.1.0";

// ─── 対話モードのラッパー ───
function InteractiveApp({
  initialConfig,
}: {
  initialConfig: ClaudexConfig;
}) {
  const [phase, setPhase] = useState<
    "onboarding" | "logging-in" | "ready" | "error"
  >(initialConfig.onboarded ? "ready" : "onboarding");
  const [config, setConfig] = useState(initialConfig);
  const [client, setClient] = useState<OpenAIClient | null>(null);
  const [email, setEmail] = useState<string | undefined>();
  const [planType, setPlanType] = useState<string | undefined>();
  const [errorMsg, setErrorMsg] = useState("");

  // 認証済みの場合はクライアント初期化
  useEffect(() => {
    if (phase === "ready" && !client) {
      initClient();
    }
  }, [phase]);

  async function initClient() {
    try {
      const auth = await ensureAuth();
      setClient(new OpenAIClient(auth.api_key, config.model));
      setEmail(auth.email);
      setPlanType(auth.plan_type);
    } catch {
      // 未ログイン — ログインフローへ
      setPhase("logging-in");
    }
  }

  async function handleOnboardingComplete(result: {
    theme: string;
    permMode: string;
    model: string;
    doLogin: boolean;
  }) {
    const updated: ClaudexConfig = {
      theme: result.theme as ClaudexConfig["theme"],
      permMode: result.permMode as ClaudexConfig["permMode"],
      model: result.model,
      onboarded: true,
      version: VERSION,
    };
    await saveConfig(updated);
    setConfig(updated);

    if (result.doLogin) {
      setPhase("logging-in");
      try {
        const auth = await login();
        setClient(new OpenAIClient(auth.api_key, updated.model));
        setEmail(auth.email);
        setPlanType(auth.plan_type);
        setPhase("ready");
      } catch (err: any) {
        setErrorMsg(`Login failed: ${err.message}`);
        setPhase("error");
      }
    } else {
      setErrorMsg("Login required. Run: claudex login");
      setPhase("error");
    }
  }

  if (phase === "onboarding") {
    return <Onboarding onComplete={handleOnboardingComplete} />;
  }

  if (phase === "logging-in") {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text bold color="cyan">claudex</Text>
        <Text> </Text>
        <Text color="yellow">Opening browser for ChatGPT login...</Text>
        <Text dimColor>Waiting for authentication callback...</Text>
      </Box>
    );
  }

  if (phase === "error") {
    return (
      <Box flexDirection="column" paddingX={2} paddingY={1}>
        <Text color="red">{errorMsg}</Text>
      </Box>
    );
  }

  if (!client) {
    return (
      <Box paddingX={2} paddingY={1}>
        <Text color="yellow">Initializing...</Text>
      </Box>
    );
  }

  return (
    <App
      client={client}
      email={email}
      planType={planType}
    />
  );
}

// ─── main ───
async function main() {
  const args = process.argv.slice(2);

  // --- login ---
  if (args[0] === "login") {
    console.log("\n  claudex login\n");
    console.log("  Opening browser for ChatGPT account login...\n");
    try {
      const auth = await login();
      console.log(`  Logged in as: ${auth.email || "unknown"}`);
      console.log(`  Plan: ${auth.plan_type || "unknown"}\n`);

      // Mark onboarded
      const config = await loadConfig();
      config.onboarded = true;
      await saveConfig(config);
    } catch (err: any) {
      console.error(`  Login failed: ${err.message}\n`);
      process.exit(1);
    }
    return;
  }

  // --- logout ---
  if (args[0] === "logout") {
    await clearAuth();
    console.log("\n  Logged out.\n");
    return;
  }

  // --- version ---
  if (args[0] === "--version" || args[0] === "-v") {
    console.log(`claudex v${VERSION}`);
    return;
  }

  // --- help ---
  if (args[0] === "--help" || args[0] === "-h" || args[0] === "help") {
    printHelp();
    return;
  }

  // --- config reset ---
  if (args[0] === "reset") {
    await clearAuth();
    const { saveConfig } = await import("./config.js");
    await saveConfig({
      theme: "dark",
      permMode: "normal",
      model: "o4-mini",
      onboarded: false,
      version: VERSION,
    });
    console.log("\n  Config and auth reset. Run claudex to set up again.\n");
    return;
  }

  // --- モデル上書き ---
  const config = await loadConfig();
  const modelIdx = args.indexOf("--model");
  const modelIdxShort = args.indexOf("-m");
  const mIdx = modelIdx !== -1 ? modelIdx : modelIdxShort;
  if (mIdx !== -1 && args[mIdx + 1]) {
    config.model = args[mIdx + 1];
  }

  // --- 非対話モード (-p) ---
  const promptIdx = args.indexOf("-p");
  if (promptIdx !== -1 && args[promptIdx + 1]) {
    await runNonInteractive(args[promptIdx + 1], config);
    return;
  }

  // --- 対話モード ---
  const { waitUntilExit } = render(
    <InteractiveApp initialConfig={config} />,
    { exitOnCtrlC: false },
  );
  await waitUntilExit();
}

async function runNonInteractive(prompt: string, config: ClaudexConfig) {
  let auth;
  try {
    auth = await ensureAuth();
  } catch {
    console.error("Not logged in. Run: claudex login");
    process.exit(1);
  }

  const client = new OpenAIClient(auth.api_key, config.model);
  const { buildSystemPrompt } = await import("./prompts/system.js");
  const { toolDefinitions, executeTool } = await import("./tools/index.js");

  const messages: any[] = [
    { role: "system", content: buildSystemPrompt() },
    { role: "user", content: prompt },
  ];

  let running = true;
  while (running) {
    let content = "";
    const toolCalls: any[] = [];

    for await (const delta of client.stream(messages, toolDefinitions)) {
      if (delta.type === "content" && delta.content) {
        process.stdout.write(delta.content);
        content += delta.content;
      }
      if (delta.type === "tool_call" && delta.tool_call) {
        toolCalls.push(delta.tool_call);
      }
      if (delta.type === "done") break;
      if (delta.type === "error") {
        console.error(delta.error);
        process.exit(1);
      }
    }
    if (content) process.stdout.write("\n");

    messages.push({
      role: "assistant",
      content: content || "",
      tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
    });

    if (toolCalls.length > 0) {
      for (const tc of toolCalls) {
        console.log(`\n[${tc.function.name}]`);
        const result = await executeTool(tc);
        if (result.error) console.error(result.output);
        messages.push({
          role: "tool",
          content: result.output,
          tool_call_id: tc.id,
        });
      }
    } else {
      running = false;
    }
  }
}

function printHelp() {
  console.log(`
  claudex v${VERSION}
  Claude Code UI + OpenAI models. No API key needed.

  Usage:
    claudex              Start interactive session (first run: setup wizard)
    claudex login        Login with ChatGPT account
    claudex logout       Clear saved credentials
    claudex reset        Reset config and auth (re-run setup)
    claudex -p "..."     Run a prompt non-interactively

  Options:
    --model, -m <model>  Set model (default: o4-mini)
    --version, -v        Show version
    --help, -h           Show help

  In-session commands:
    /clear               Clear conversation
    /model <name>        Change model
    /auto                Toggle auto-approve tools
    /exit                Exit

  Supported models:
    o4-mini              Fast, cost-effective (default)
    gpt-4.1              Balanced performance
    gpt-4.1-mini         Lightweight
    gpt-4.1-nano         Fastest
    o3                   Most capable reasoning
    o3-pro               Maximum quality
`);
}

main().catch((err) => {
  console.error(err.message || err);
  process.exit(1);
});
