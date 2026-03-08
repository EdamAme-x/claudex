# claudex

Claude Code UI + OpenAI models. ChatGPT account login — no API key needed.

## Features

- Claude Code-like interactive TUI (built with Ink/React)
- Uses OpenAI models (o4-mini, gpt-4.1, o3, o3-pro, etc.)
- Built-in tools: Bash, Read, Write, Edit, Glob, Grep
- Setup wizard on first run (theme, permission mode, model selection)
- One-shot mode with `-p` flag

## Install

### macOS

**Homebrew (coming soon)**

**Installer (.pkg)**

Download `claudex-*-macos-arm64.pkg` (Apple Silicon) or `claudex-*-macos-x64.pkg` (Intel) from [Releases](../../releases) and double-click to install.

**DMG**

Download the `.dmg`, open it, and run `install.command`.

**tar.gz**

```bash
tar xzf claudex-*-macos-*.tar.gz
cd claudex-*
sudo cp claudex /usr/local/bin/
```

### Linux

**Debian / Ubuntu (.deb)**

```bash
sudo dpkg -i claudex-*-linux-amd64.deb
```

**Fedora / RHEL (.rpm)**

```bash
sudo rpm -i claudex-*-linux-*.rpm
```

**AppImage**

```bash
chmod +x claudex-*-linux-*.AppImage
./claudex-*-linux-*.AppImage
```

**tar.gz**

```bash
tar xzf claudex-*-linux-*.tar.gz
cd claudex-*
./install.sh
```

### Windows

**Installer (.exe)**

Download `claudex-*-windows-x64-setup.exe` from [Releases](../../releases) and run it.

**ZIP**

```powershell
Expand-Archive claudex-*-windows-x64.zip -DestinationPath .
.\claudex\claudex.exe
```

Add `claudex.exe` to your PATH for global access.

### Build from source

```bash
git clone https://github.com/serkenn/claudex-mac.git
cd claudex-mac
npm install
npm run dev
```

## Usage

```bash
claudex              # Start interactive session
claudex login        # Login with ChatGPT account
claudex logout       # Clear credentials
claudex reset        # Reset config and re-run setup
claudex -p "..."     # One-shot prompt
claudex --model o3   # Use specific model
```

### In-session commands

| Command        | Description           |
|----------------|-----------------------|
| `/clear`       | Clear conversation    |
| `/model <name>`| Change model          |
| `/auto`        | Toggle auto-approve   |
| `/exit`        | Exit                  |

### Supported models

| Model          | Description                    |
|----------------|--------------------------------|
| `o4-mini`      | Fast, cost-effective (default) |
| `gpt-4.1`     | Balanced performance           |
| `gpt-4.1-mini` | Lightweight                   |
| `gpt-4.1-nano` | Fastest                       |
| `o3`           | Most capable reasoning         |
| `o3-pro`       | Maximum quality                |

## License

MIT
