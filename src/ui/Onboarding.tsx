/**
 * Onboarding - 初回起動時のセットアップUI（Claude Code風）
 */
import React, { useState, useEffect } from "react";
import { Box, Text, useInput, useApp } from "ink";

type Theme = "dark" | "light" | "auto";
type PermMode = "normal" | "auto-approve" | "plan-mode";

interface OnboardingResult {
  theme: Theme;
  permMode: PermMode;
  model: string;
  doLogin: boolean;
}

interface Props {
  onComplete: (result: OnboardingResult) => void;
}

type Step = "welcome" | "theme" | "permissions" | "model" | "login";

const MODELS = [
  { id: "o4-mini", label: "o4-mini", desc: "Fast, cost-effective" },
  { id: "gpt-4.1", label: "GPT-4.1", desc: "Balanced performance" },
  { id: "gpt-4.1-mini", label: "GPT-4.1 Mini", desc: "Lightweight" },
  { id: "gpt-4.1-nano", label: "GPT-4.1 Nano", desc: "Fastest, cheapest" },
  { id: "o3", label: "o3", desc: "Most capable reasoning" },
  { id: "o3-pro", label: "o3-pro", desc: "Maximum quality (slow)" },
];

export function Onboarding({ onComplete }: Props) {
  const [step, setStep] = useState<Step>("welcome");
  const [theme, setTheme] = useState<Theme>("dark");
  const [permMode, setPermMode] = useState<PermMode>("normal");
  const [modelIdx, setModelIdx] = useState(0);
  const [selectedTheme, setSelectedTheme] = useState(0);
  const [selectedPerm, setSelectedPerm] = useState(0);

  // Welcome screen
  if (step === "welcome") {
    return (
      <WelcomeStep onContinue={() => setStep("theme")} />
    );
  }

  // Theme selection
  if (step === "theme") {
    return (
      <ThemeStep
        selected={selectedTheme}
        onSelect={setSelectedTheme}
        onConfirm={() => {
          setTheme((["dark", "light", "auto"] as Theme[])[selectedTheme]);
          setStep("permissions");
        }}
      />
    );
  }

  // Permission mode
  if (step === "permissions") {
    return (
      <PermStep
        selected={selectedPerm}
        onSelect={setSelectedPerm}
        onConfirm={() => {
          setPermMode(
            (["normal", "auto-approve", "plan-mode"] as PermMode[])[selectedPerm],
          );
          setStep("model");
        }}
      />
    );
  }

  // Model selection
  if (step === "model") {
    return (
      <ModelStep
        models={MODELS}
        selected={modelIdx}
        onSelect={setModelIdx}
        onConfirm={() => setStep("login")}
      />
    );
  }

  // Login prompt
  return (
    <LoginStep
      onConfirm={(doLogin) => {
        onComplete({
          theme,
          permMode,
          model: MODELS[modelIdx].id,
          doLogin,
        });
      }}
    />
  );
}

// ─── Welcome ───
function WelcomeStep({ onContinue }: { onContinue: () => void }) {
  useInput((input, key) => {
    if (key.return) onContinue();
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Box flexDirection="column" alignItems="center">
        <Text bold color="cyan">
          {`
   _____ _                 _
  / ____| |               | |
 | |    | | __ _ _   _  __| | _____  __
 | |    | |/ _\` | | | |/ _\` |/ _ \\ \\/ /
 | |____| | (_| | |_| | (_| |  __/>  <
  \\_____|_|\\__,_|\\__,_|\\__,_|\\___/_/\\_\\
          `}
        </Text>
        <Text> </Text>
        <Text bold>Welcome to claudex v0.1.0</Text>
        <Text dimColor>Claude Code UI + OpenAI Models</Text>
        <Text> </Text>
        <Text dimColor>ChatGPT account login only. No API key needed.</Text>
      </Box>

      <Box marginTop={1} justifyContent="center">
        <Text>
          Press <Text color="cyan" bold>Enter</Text> to set up
        </Text>
      </Box>
    </Box>
  );
}

// ─── Theme ───
function ThemeStep({
  selected,
  onSelect,
  onConfirm,
}: {
  selected: number;
  onSelect: (n: number) => void;
  onConfirm: () => void;
}) {
  const options = [
    { label: "Dark", desc: "Dark background (recommended)" },
    { label: "Light", desc: "Light background" },
    { label: "Auto", desc: "Match terminal theme" },
  ];

  useInput((input, key) => {
    if (key.upArrow) onSelect(Math.max(0, selected - 1));
    if (key.downArrow) onSelect(Math.min(options.length - 1, selected + 1));
    if (key.return) onConfirm();
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Text bold color="cyan">Theme</Text>
      <Text dimColor>Choose your preferred color scheme</Text>
      <Text> </Text>
      {options.map((opt, i) => (
        <Box key={i}>
          <Text color={i === selected ? "cyan" : undefined}>
            {i === selected ? "❯ " : "  "}
            {opt.label}
          </Text>
          <Text dimColor> — {opt.desc}</Text>
        </Box>
      ))}
      <Text> </Text>
      <Text dimColor>↑↓ to navigate, Enter to select</Text>
    </Box>
  );
}

// ─── Permissions ───
function PermStep({
  selected,
  onSelect,
  onConfirm,
}: {
  selected: number;
  onSelect: (n: number) => void;
  onConfirm: () => void;
}) {
  const options = [
    {
      label: "Normal",
      desc: "Ask before running Bash, Write, Edit",
    },
    {
      label: "Auto-approve",
      desc: "Run all tools without asking (YOLO mode)",
    },
    {
      label: "Plan mode",
      desc: "Show plan before executing, confirm once",
    },
  ];

  useInput((input, key) => {
    if (key.upArrow) onSelect(Math.max(0, selected - 1));
    if (key.downArrow) onSelect(Math.min(options.length - 1, selected + 1));
    if (key.return) onConfirm();
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Text bold color="cyan">Permission Mode</Text>
      <Text dimColor>How should claudex handle tool execution?</Text>
      <Text> </Text>
      {options.map((opt, i) => (
        <Box key={i} flexDirection="column">
          <Box>
            <Text color={i === selected ? "cyan" : undefined}>
              {i === selected ? "❯ " : "  "}
              <Text bold>{opt.label}</Text>
            </Text>
          </Box>
          <Box paddingLeft={4}>
            <Text dimColor>{opt.desc}</Text>
          </Box>
        </Box>
      ))}
      <Text> </Text>
      <Text dimColor>↑↓ to navigate, Enter to select</Text>
    </Box>
  );
}

// ─── Model ───
function ModelStep({
  models,
  selected,
  onSelect,
  onConfirm,
}: {
  models: { id: string; label: string; desc: string }[];
  selected: number;
  onSelect: (n: number) => void;
  onConfirm: () => void;
}) {
  useInput((input, key) => {
    if (key.upArrow) onSelect(Math.max(0, selected - 1));
    if (key.downArrow) onSelect(Math.min(models.length - 1, selected + 1));
    if (key.return) onConfirm();
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Text bold color="cyan">Model</Text>
      <Text dimColor>Choose your default OpenAI model</Text>
      <Text> </Text>
      {models.map((m, i) => (
        <Box key={i}>
          <Text color={i === selected ? "cyan" : undefined}>
            {i === selected ? "❯ " : "  "}
            <Text bold>{m.label.padEnd(16)}</Text>
          </Text>
          <Text dimColor>{m.desc}</Text>
        </Box>
      ))}
      <Text> </Text>
      <Text dimColor>↑↓ to navigate, Enter to select. Change anytime with /model</Text>
    </Box>
  );
}

// ─── Login ───
function LoginStep({
  onConfirm,
}: {
  onConfirm: (doLogin: boolean) => void;
}) {
  const [selected, setSelected] = useState(0);

  useInput((input, key) => {
    if (key.upArrow || key.downArrow) setSelected(selected === 0 ? 1 : 0);
    if (key.return) onConfirm(selected === 0);
  });

  return (
    <Box flexDirection="column" paddingX={2} paddingY={1}>
      <Text bold color="cyan">Login</Text>
      <Text dimColor>
        claudex uses your ChatGPT account. No API key needed.
      </Text>
      <Text> </Text>
      <Box>
        <Text color={selected === 0 ? "cyan" : undefined}>
          {selected === 0 ? "❯ " : "  "}
          <Text bold>Login with ChatGPT account</Text>
        </Text>
      </Box>
      <Box paddingLeft={4}>
        <Text dimColor>Opens browser for secure OAuth login</Text>
      </Box>
      <Box>
        <Text color={selected === 1 ? "cyan" : undefined}>
          {selected === 1 ? "❯ " : "  "}
          <Text bold>Skip for now</Text>
        </Text>
      </Box>
      <Box paddingLeft={4}>
        <Text dimColor>You can login later with: claudex login</Text>
      </Box>
      <Text> </Text>
      <Text dimColor>Enter to confirm</Text>
    </Box>
  );
}
