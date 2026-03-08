#!/usr/bin/env bash
set -euo pipefail

# claudex 全プラットフォームバイナリビルド (bun compile)
# Bunのクロスコンパイル機能でワンショットビルド

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT_DIR/dist/bin"

cd "$ROOT_DIR"

mkdir -p "$BIN_DIR"

# Bunのクロスコンパイルターゲット
# https://bun.sh/docs/bundler/executables#cross-compile
TARGETS=(
  "bun-linux-x64"
  "bun-linux-arm64"
  "bun-darwin-x64"
  "bun-darwin-arm64"
  "bun-windows-x64"
  # "bun-windows-arm64"  # Bun doesn't support win-arm64 compile target yet
)

echo "=== Building claudex binaries ==="
echo ""

for target in "${TARGETS[@]}"; do
  os=$(echo "$target" | cut -d'-' -f2)
  arch=$(echo "$target" | cut -d'-' -f3)

  if [ "$os" = "windows" ]; then
    outname="claudex-${os}-${arch}.exe"
  else
    outname="claudex-${os}-${arch}"
  fi

  echo "Building: ${outname} (target: ${target})"

  bun build src/index.tsx \
    --compile \
    --target="$target" \
    --outfile="$BIN_DIR/$outname" \
    2>&1 || echo "  WARNING: Failed to build $target"
done

echo ""
echo "=== Built binaries ==="
ls -lh "$BIN_DIR/"
