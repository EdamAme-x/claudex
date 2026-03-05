#!/usr/bin/env bun

import { $ } from "bun";
import { mkdtempSync, mkdirSync, rmSync, copyFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

const targets = [
  { target: "bun-linux-x64-modern", output: "claudex-linux-x64" },
  { target: "bun-linux-arm64", output: "claudex-linux-arm64" },
  { target: "bun-darwin-x64", output: "claudex-macos-x64" },
  { target: "bun-darwin-arm64", output: "claudex-macos-arm64" },
  { target: "bun-windows-x64", output: "claudex-windows-x64.exe" },
];

await $`mkdir -p dist`;

for (const { target, output } of targets) {
  console.log(`[build] ${output} (${target})`);
  await $`bun build ./src/claudex.ts --compile --target ${target} --outfile ./dist/${output}`;
}

if (process.platform === "darwin") {
  const version = (process.env.CLAUDEX_PKG_VERSION || "1.0.0").trim();
  const staging = mkdtempSync(join(tmpdir(), "claudex-pkg-"));

  try {
    for (const arch of ["arm64", "x64"] as const) {
      const pkgRoot = join(staging, arch, "root");
      const installBinDir = join(pkgRoot, "usr", "local", "bin");
      mkdirSync(installBinDir, { recursive: true });

      const sourceBinary = join("dist", `claudex-macos-${arch}`);
      const stagedBinary = join(installBinDir, "claudex");
      copyFileSync(sourceBinary, stagedBinary);
      await $`chmod 755 ${stagedBinary}`;

      const pkgOutput = `./dist/claudex-macos-${arch}.pkg`;
      await $`/usr/bin/env COPYFILE_DISABLE=1 pkgbuild --root ${pkgRoot} --identifier io.github.edamamex.claudex.${arch} --version ${version} --install-location / --filter \\.DS_Store$ --filter /\\._ ${pkgOutput}`;
      console.log(`[pkg] claudex-macos-${arch}.pkg (version=${version})`);
    }
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

await $`ls -lah dist`;
