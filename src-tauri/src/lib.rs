mod audio;
mod config;
mod indicator;
mod input;
mod model;
mod native_shortcut;
mod shortcut;
mod state;
mod tray;
mod whisper;

use std::sync::{Arc, Mutex};

use config::AppConfig;
use tauri::Manager;
use state::{AppState, SharedAppState};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    let config = AppConfig::load().unwrap_or_default();
    let mut app_state_inner = AppState::new(config.clone());

    // Load model at startup if one is selected
    if let Some(ref model_id) = config.selected_model {
        log::info!("Loading previously selected model: {}", model_id);
        if let Ok(models_dir) = AppConfig::models_dir() {
            let model_path = models_dir.join(format!("ggml-{}.bin", model_id));
            if model_path.exists() {
                match whisper::WhisperEngine::from_file(&model_path) {
                    Ok(engine) => {
                        app_state_inner.whisper_engine = Some(std::sync::Arc::new(engine));
                        log::info!("Model {} loaded successfully at startup", model_id);
                    }
                    Err(e) => {
                        log::error!("Failed to load model at startup: {}", e);
                    }
                }
            }
        }
    }

    let app_state: SharedAppState = Arc::new(Mutex::new(app_state_inner));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(app_state.clone())
        .setup(move |app| {

            tray::create_tray(app)?;

            // Set indicator window level to appear above fullscreen apps (must be on main thread)
            #[cfg(target_os = "macos")]
            indicator::setup_indicator_window(app.handle());

            // Request microphone permission at startup
            #[cfg(target_os = "macos")]
            {
                log::info!("Checking microphone permission at startup");
                native_shortcut::request_microphone_permission();
            }

            let app_handle = app.handle().clone();
            let state = app_state.clone();

            // Always start native keyboard listener
            // It checks the current shortcut on each keypress and only acts if it's a native shortcut
            log::info!("Starting native keyboard listener");
            native_shortcut::start_native_listener(app_handle.clone(), state.clone());

            // Also register global shortcut if current shortcut is not native
            let is_native = {
                if let Ok(s) = state.lock() {
                    native_shortcut::is_native_shortcut(&s.config.shortcut)
                } else {
                    false
                }
            };

            if !is_native {
                // Register global shortcut for key combinations
                std::thread::spawn(move || {
                    if let Err(e) = shortcut::register_default_shortcut(&app_handle, state.clone())
                    {
                        log::error!("Failed to register shortcut: {}", e);
                    } else {
                        // Track the registered shortcut
                        if let Ok(mut s) = state.lock() {
                            s.active_shortcut = Some(s.config.shortcut.clone());
                        }
                    }
                });
            }

            // Add window close handler - hide instead of quit (Spokenly-style)
            if let Some(window) = app.get_webview_window("main") {
                let window_clone = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = window_clone.hide();
                    }
                });

                // Show settings on first launch (no model selected) so user can set up
                if config.selected_model.is_none() {
                    log::info!("No model selected - showing settings for first-time setup");
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            config::get_config,
            config::set_config,
            model::list_available_models,
            model::list_downloaded_models,
            model::download_model,
            model::delete_model,
            model::get_download_progress,
            state::get_app_state,
            state::toggle_recording,
            input::check_accessibility_permission,
            shortcut::unregister_current_shortcut,
            shortcut::register_current_shortcut,
            native_shortcut::pause_native_listener,
            native_shortcut::resume_native_listener,
            native_shortcut::check_accessibility,
            native_shortcut::check_microphone,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
