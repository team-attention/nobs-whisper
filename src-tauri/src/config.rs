use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    /// Selected model ID (e.g., "small", "large-v3")
    pub selected_model: Option<String>,
    /// Global shortcut (e.g., "Ctrl+Alt+Space")
    pub shortcut: String,
    /// Speech recognition language (e.g., "ko", "en", "auto")
    pub language: String,
    /// Auto-launch on system startup
    pub auto_launch: bool,
    /// Maximum recording duration in seconds (0 = unlimited, capped at 600s)
    #[serde(default = "default_max_recording_duration")]
    pub max_recording_duration: u64,
    /// Custom vocabulary to help Whisper recognize specific terms
    #[serde(default = "default_custom_vocabulary")]
    pub custom_vocabulary: String,
    /// Push-to-talk mode: record while key is held, stop when released
    #[serde(default)]
    pub push_to_talk: bool,
}

fn default_max_recording_duration() -> u64 {
    60
}

fn default_custom_vocabulary() -> String {
    "Claude Code, Anthropic, Supabase, Vercel, shadcn, tRPC, Drizzle, Zod, pnpm, Bun, Deno, Turso, Neon, PlanetScale, Turborepo, Tauri, SvelteKit, Nuxt, Astro, Vite, Zustand, TanStack, LangChain, LlamaIndex, Ollama, Cursor, Neovim, Vitest, Playwright, Prisma, Radix, Fly.io, Railway, Cloudflare Workers, Hono, htmx, Biome, oxlint, Rspack, Turbopack, Qwik, SolidJS, Convex, Upstash, Resend, Inngest, Replit, v0, Lovable, Bolt, WindSurf, Codeium, Supermaven, Aider, OpenRouter, Perplexity, Groq, Mistral, Cohere, Replicate".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            selected_model: None,
            shortcut: "Ctrl+Alt+Space".to_string(),
            language: "auto".to_string(),
            auto_launch: false,
            max_recording_duration: default_max_recording_duration(),
            custom_vocabulary: default_custom_vocabulary(),
            push_to_talk: false,
        }
    }
}

impl AppConfig {
    fn config_dir() -> Result<PathBuf, ConfigError> {
        let dir = dirs::config_dir()
            .ok_or_else(|| {
                ConfigError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Config directory not found",
                ))
            })?
            .join("NobsWhisper");

        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }

        Ok(dir)
    }

    fn config_path() -> Result<PathBuf, ConfigError> {
        Ok(Self::config_dir()?.join("config.json"))
    }

    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path()?;
        if !path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let content = fs::read_to_string(&path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path()?;
        let content = serde_json::to_string_pretty(self)?;
        fs::write(&path, content)?;
        Ok(())
    }

    pub fn models_dir() -> Result<PathBuf, ConfigError> {
        let dir = Self::config_dir()?.join("models");
        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }
        Ok(dir)
    }
}

#[tauri::command]
pub fn get_config() -> Result<AppConfig, String> {
    AppConfig::load().map_err(|e| e.to_string())
}

#[tauri::command]
pub fn set_config(
    config: AppConfig,
    state: tauri::State<'_, crate::state::SharedAppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    // Save config to disk
    config.save().map_err(|e| e.to_string())?;

    // Update app state and load model if changed
    let mut app_state = state.lock().map_err(|e| e.to_string())?;

    // Check if selected model changed
    let model_changed = app_state.config.selected_model != config.selected_model;

    // Check if shortcut changed
    let shortcut_changed = app_state.config.shortcut != config.shortcut;
    let new_shortcut = config.shortcut.clone();
    let old_shortcut = app_state.active_shortcut.clone();

    // Update config in state
    app_state.config = config.clone();

    // Load model if changed and a model is selected
    if model_changed {
        if let Some(ref model_id) = config.selected_model {
            log::info!("Loading model: {}", model_id);

            // Get model path
            let models_dir = AppConfig::models_dir().map_err(|e| e.to_string())?;
            let model_path = models_dir.join(format!("ggml-{}.bin", model_id));

            if model_path.exists() {
                match crate::whisper::WhisperEngine::from_file(&model_path) {
                    Ok(engine) => {
                        app_state.whisper_engine = Some(std::sync::Arc::new(engine));
                        log::info!("Model {} loaded successfully", model_id);
                    }
                    Err(e) => {
                        log::error!("Failed to load model: {}", e);
                        app_state.error = Some(format!("Failed to load model: {}", e));
                    }
                }
            } else {
                log::warn!("Model file not found: {:?}", model_path);
            }
        } else {
            // No model selected, clear engine
            app_state.whisper_engine = None;
        }
    }

    // Update shortcut if changed
    if shortcut_changed {
        // Release lock before shortcut operations
        drop(app_state);

        let new_is_native = crate::native_shortcut::is_native_shortcut(&new_shortcut);

        // Unregister old global shortcut if it was a standard shortcut
        if let Some(ref old) = old_shortcut {
            if !crate::native_shortcut::is_native_shortcut(old) {
                if let Err(e) = crate::shortcut::unregister_shortcut(&app, old) {
                    log::warn!("Failed to unregister old shortcut: {}", e);
                }
            }
        }

        // Only register global shortcut if new shortcut is NOT native
        // Native shortcuts are handled by the always-running native listener
        if !new_is_native {
            if let Err(e) = crate::shortcut::register_shortcut(&app, &new_shortcut) {
                return Err(format!("Failed to register shortcut: {}", e));
            }
        }

        // Update active shortcut in state
        let mut app_state = state.lock().map_err(|e| e.to_string())?;
        app_state.active_shortcut = if new_is_native {
            None // Native shortcuts don't use Tauri's global shortcut system
        } else {
            Some(new_shortcut)
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.shortcut, "Ctrl+Alt+Space");
        assert_eq!(config.language, "auto");
        assert!(!config.auto_launch);
        assert!(config.selected_model.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: AppConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.shortcut, parsed.shortcut);
    }
}
