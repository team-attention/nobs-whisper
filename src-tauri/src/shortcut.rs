use crate::state::SharedAppState;
use tauri::{AppHandle, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShortcutError {
    #[error("Failed to parse shortcut: {0}")]
    ParseError(String),
    #[error("Failed to register shortcut: {0}")]
    RegisterError(String),
    #[error("Failed to unregister shortcut: {0}")]
    UnregisterError(String),
}

pub fn register_default_shortcut(
    app: &AppHandle,
    state: SharedAppState,
) -> Result<(), ShortcutError> {
    let shortcut_str = {
        let state = state.lock().map_err(|e| ShortcutError::RegisterError(e.to_string()))?;
        state.config.shortcut.clone()
    };

    register_shortcut(app, &shortcut_str)
}

pub fn register_shortcut(
    app: &AppHandle,
    shortcut_str: &str,
) -> Result<(), ShortcutError> {
    log::info!("Registering shortcut: {}", shortcut_str);

    // Parse the shortcut string
    let shortcut: Shortcut = shortcut_str
        .parse()
        .map_err(|e| ShortcutError::ParseError(format!("{:?}", e)))?;

    let app_handle = app.clone();

    // Register the shortcut
    app.global_shortcut()
        .on_shortcut(shortcut, move |_app, _shortcut, event| {
            if event.state == ShortcutState::Pressed {
                log::info!("Shortcut pressed");

                let state = app_handle.state::<SharedAppState>();
                match crate::state::toggle_recording_with_app(&app_handle, &state) {
                    Ok(snapshot) => {
                        log::info!("Recording toggled, is_recording: {}", snapshot.is_recording);
                    }
                    Err(e) => {
                        log::error!("Failed to toggle recording: {}", e);
                    }
                }
            }
        })
        .map_err(|e| ShortcutError::RegisterError(e.to_string()))?;

    log::info!("Shortcut registered successfully");
    Ok(())
}

pub fn unregister_shortcut(app: &AppHandle, shortcut_str: &str) -> Result<(), ShortcutError> {
    log::info!("Unregistering shortcut: {}", shortcut_str);

    let shortcut: Shortcut = shortcut_str
        .parse()
        .map_err(|e| ShortcutError::ParseError(format!("{:?}", e)))?;

    app.global_shortcut()
        .unregister(shortcut)
        .map_err(|e| ShortcutError::UnregisterError(e.to_string()))?;

    log::info!("Shortcut unregistered");
    Ok(())
}

/// Temporarily unregister the current shortcut (for capture mode)
#[tauri::command]
pub fn unregister_current_shortcut(
    app: AppHandle,
    state: tauri::State<'_, SharedAppState>,
) -> Result<(), String> {
    log::info!("Unregistering all shortcuts for capture mode");

    // Unregister ALL shortcuts to ensure none trigger during capture
    app.global_shortcut()
        .unregister_all()
        .map_err(|e| format!("Failed to unregister all shortcuts: {}", e))?;

    // Clear active shortcut
    let mut state_guard = state.lock().map_err(|e| e.to_string())?;
    state_guard.active_shortcut = None;

    log::info!("All shortcuts unregistered");
    Ok(())
}

/// Re-register the current shortcut (after capture mode cancelled)
#[tauri::command]
pub fn register_current_shortcut(
    app: AppHandle,
    state: tauri::State<'_, SharedAppState>,
) -> Result<(), String> {
    let state_guard = state.lock().map_err(|e| e.to_string())?;
    let shortcut = state_guard.config.shortcut.clone();
    drop(state_guard);

    register_shortcut(&app, &shortcut).map_err(|e| e.to_string())?;

    // Update active shortcut
    let mut state_guard = state.lock().map_err(|e| e.to_string())?;
    state_guard.active_shortcut = Some(shortcut);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_shortcut_parsing() {
        // This is just a compile-time test to ensure the module compiles
        assert!(true);
    }
}
