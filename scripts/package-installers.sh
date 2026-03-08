#!/usr/bin/env bash
set -euo pipefail

# claudex インストーラーパッケージ作成
# バイナリ名: claudex-{os}-{arch}[.exe]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT_DIR/dist/bin"
PKG_DIR="$ROOT_DIR/dist/packages"
VERSION="${VERSION:-0.1.0}"

mkdir -p "$PKG_DIR"

find_bin() {
  local os="$1" arch="$2"
  local ext=""
  [ "$os" = "windows" ] && ext=".exe"
  local bin="$BIN_DIR/claudex-${os}-${arch}${ext}"
  [ -f "$bin" ] && echo "$bin" || echo ""
}

# ─── Linux tar.gz ───
linux_tar() {
  local arch="$1"
  local bin=$(find_bin linux "$arch")
  [ -z "$bin" ] && return
  echo "  tar.gz: linux-${arch}"
  local s=$(mktemp -d)
  mkdir -p "$s/claudex-${VERSION}"
  cp "$bin" "$s/claudex-${VERSION}/claudex"
  chmod +x "$s/claudex-${VERSION}/claudex"
  cat > "$s/claudex-${VERSION}/install.sh" << 'EOF'
#!/bin/bash
set -e
sudo cp claudex /usr/local/bin/claudex && sudo chmod +x /usr/local/bin/claudex
echo "claudex installed! Run: claudex"
EOF
  chmod +x "$s/claudex-${VERSION}/install.sh"
  tar -czf "$PKG_DIR/claudex-${VERSION}-linux-${arch}.tar.gz" -C "$s" "claudex-${VERSION}"
  rm -rf "$s"
}

# ─── Linux .deb ───
linux_deb() {
  local arch="$1"
  local bin=$(find_bin linux "$arch")
  [ -z "$bin" ] && return
  command -v dpkg-deb &>/dev/null || { echo "  skip .deb (no dpkg-deb)"; return; }
  local deb_arch="$arch"
  [ "$arch" = "x64" ] && deb_arch="amd64"
  echo "  deb: linux-${deb_arch}"
  local s=$(mktemp -d)
  mkdir -p "$s/DEBIAN" "$s/usr/local/bin"
  cp "$bin" "$s/usr/local/bin/claudex" && chmod +x "$s/usr/local/bin/claudex"
  cat > "$s/DEBIAN/control" << EOF
Package: claudex
Version: ${VERSION}
Section: devel
Priority: optional
Architecture: ${deb_arch}
Maintainer: claudex <claudex@users.noreply.github.com>
Description: Claude Code UI + OpenAI models CLI
 Interactive coding assistant. No API key needed.
EOF
  dpkg-deb --build "$s" "$PKG_DIR/claudex-${VERSION}-linux-${deb_arch}.deb"
  rm -rf "$s"
}

# ─── Linux .rpm ───
linux_rpm() {
  local arch="$1"
  local bin=$(find_bin linux "$arch")
  [ -z "$bin" ] && return
  command -v rpmbuild &>/dev/null || { echo "  skip .rpm (no rpmbuild)"; return; }
  local rpm_arch="$arch"
  [ "$arch" = "x64" ] && rpm_arch="x86_64"
  [ "$arch" = "arm64" ] && rpm_arch="aarch64"
  echo "  rpm: linux-${rpm_arch}"
  local s=$(mktemp -d)
  mkdir -p "$s"/{BUILD,RPMS,SOURCES,SPECS,SRPMS,BUILDROOT/usr/local/bin}
  cp "$bin" "$s/BUILDROOT/usr/local/bin/claudex" && chmod +x "$s/BUILDROOT/usr/local/bin/claudex"
  cat > "$s/SPECS/claudex.spec" << EOF
Name: claudex
Version: ${VERSION}
Release: 1
Summary: Claude Code UI + OpenAI models CLI
License: MIT
AutoReqProv: no
%description
Interactive coding assistant.
%install
mkdir -p %{buildroot}/usr/local/bin
cp %{_topdir}/BUILDROOT/usr/local/bin/claudex %{buildroot}/usr/local/bin/
%files
/usr/local/bin/claudex
EOF
  rpmbuild --define "_topdir $s" -bb "$s/SPECS/claudex.spec" && cp "$s"/RPMS/*/*.rpm "$PKG_DIR/" || true
  rm -rf "$s"
}

# ─── macOS .pkg ───
macos_pkg() {
  local arch="$1"
  local bin=$(find_bin darwin "$arch")
  [ -z "$bin" ] && return
  command -v pkgbuild &>/dev/null || { echo "  skip .pkg (no pkgbuild)"; return; }
  echo "  pkg: macos-${arch}"
  local s=$(mktemp -d)
  mkdir -p "$s/usr/local/bin"
  cp "$bin" "$s/usr/local/bin/claudex" && chmod +x "$s/usr/local/bin/claudex"
  pkgbuild --root "$s" --identifier com.claudex.cli --version "$VERSION" \
    --install-location / "$PKG_DIR/claudex-${VERSION}-macos-${arch}.pkg"
  rm -rf "$s"
}

# ─── macOS .dmg ───
macos_dmg() {
  local arch="$1"
  local bin=$(find_bin darwin "$arch")
  [ -z "$bin" ] && return
  command -v hdiutil &>/dev/null || { echo "  skip .dmg (no hdiutil)"; return; }
  echo "  dmg: macos-${arch}"
  local s=$(mktemp -d)
  cp "$bin" "$s/claudex" && chmod +x "$s/claudex"
  cat > "$s/install.command" << 'EOF'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
sudo cp "$DIR/claudex" /usr/local/bin/claudex
sudo chmod +x /usr/local/bin/claudex
echo "claudex installed! Run: claudex"
read -p "Press Enter to close..."
EOF
  chmod +x "$s/install.command"
  hdiutil create -volname "claudex-${VERSION}" -srcfolder "$s" \
    -ov -format UDZO "$PKG_DIR/claudex-${VERSION}-macos-${arch}.dmg"
  rm -rf "$s"
}

# ─── macOS tar.gz ───
macos_tar() {
  local arch="$1"
  local bin=$(find_bin darwin "$arch")
  [ -z "$bin" ] && return
  echo "  tar.gz: macos-${arch}"
  local s=$(mktemp -d)
  mkdir -p "$s/claudex-${VERSION}"
  cp "$bin" "$s/claudex-${VERSION}/claudex" && chmod +x "$s/claudex-${VERSION}/claudex"
  tar -czf "$PKG_DIR/claudex-${VERSION}-macos-${arch}.tar.gz" -C "$s" "claudex-${VERSION}"
  rm -rf "$s"
}

# ─── Windows .zip ───
windows_zip() {
  local arch="$1"
  local bin=$(find_bin windows "$arch")
  [ -z "$bin" ] && return
  echo "  zip: windows-${arch}"
  local s=$(mktemp -d)
  mkdir -p "$s/claudex"
  cp "$bin" "$s/claudex/claudex.exe"
  cat > "$s/claudex/install.bat" << 'BAT'
@echo off
echo Installing claudex...
mkdir "%LOCALAPPDATA%\claudex\bin" 2>nul
copy /Y claudex.exe "%LOCALAPPDATA%\claudex\bin\claudex.exe"
setx PATH "%PATH%;%LOCALAPPDATA%\claudex\bin" >nul 2>&1
echo Done! Open a new terminal and run: claudex
pause
BAT
  (cd "$s" && zip -qr "$PKG_DIR/claudex-${VERSION}-windows-${arch}.zip" claudex/) || \
    (cd "$s" && tar -czf "$PKG_DIR/claudex-${VERSION}-windows-${arch}.zip" claudex/)
  rm -rf "$s"
}

# ─── Run all ───
echo "=== Packaging claudex v${VERSION} ==="
echo ""

echo "Linux:"
for a in x64 arm64; do linux_tar "$a"; linux_deb "$a"; linux_rpm "$a"; done

echo "macOS:"
for a in x64 arm64; do macos_pkg "$a"; macos_dmg "$a"; macos_tar "$a"; done

echo "Windows:"
for a in x64 arm64; do windows_zip "$a"; done

echo ""
echo "=== Packages ==="
ls -lh "$PKG_DIR/" 2>/dev/null || echo "No packages"
