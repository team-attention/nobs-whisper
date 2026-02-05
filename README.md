# Nobs Whisper

Free, local speech-to-text for macOS. No subscriptions, no API keys, no BS.

Tired of wrapper apps charging $10/month for OpenAI's **free** Whisper model? Same.

## Download

**[Download Latest Release](https://github.com/team-attention/nobs-whisper/releases/latest)**

## How to Use

1. Open Nobs Whisper and go to **Settings**
2. Download a Whisper model (start with `base` or `small`)
3. Set your preferred hotkey
4. Press the hotkey, speak, press again
5. Text appears where your cursor is

That's it. Everything runs locally on your Mac. Your voice data never leaves your machine.

## Features

- **100% Local** - Uses OpenAI Whisper models running on your device
- **Metal GPU Acceleration** - Fast transcription on Apple Silicon
- **Global Hotkey** - Works anywhere, even in fullscreen apps
- **Left/Right Key Detection** - Use RightOption, LeftCmd, etc. as single-key shortcuts
- **Auto-paste** - Types directly into focused input, or copies to clipboard if none
- **Multi-language** - Supports Korean, English, Japanese, Chinese, and more
- **Custom Vocabulary** - Help Whisper recognize technical terms like "Supabase", "Claude Code", etc.

## Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 75MB | Fastest | Basic |
| base | 142MB | Fast | Good |
| small | 466MB | Medium | Better |
| medium | 1.5GB | Slow | Great |
| large-v3 | 3GB | Slowest | Best |
| large-v3-turbo | 1.6GB | Medium | Great |

Download models directly from the app. Start with `base` or `small`.

## Requirements

- macOS 10.15+
- Apple Silicon recommended (Metal acceleration)
- Microphone permission
- Accessibility permission (for typing text)

## Build

```bash
# Install dependencies
npm install

# Run in development
npm run tauri dev

# Build for production
npm run tauri build
```

## Tech Stack

- [Tauri](https://tauri.app/) - Rust + Web frontend
- [whisper-rs](https://codeberg.org/tazz4843/whisper-rs) - Whisper bindings for Rust
- [SvelteKit](https://kit.svelte.dev/) - Frontend
- Native Swift helper for floating indicator

## License

MIT
# test trigger 2026년  2월  5일 목요일 11시 11분 15초 KST
