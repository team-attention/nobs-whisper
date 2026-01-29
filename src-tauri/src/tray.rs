use crate::state::SharedAppState;
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::menu::{Menu, MenuItem};
use tauri::{App, AppHandle, Manager};

pub fn create_tray(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    let record_item = MenuItem::with_id(app, "record", "Start Recording", true, None::<&str>)?;
    let settings_item = MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit Nobs Whisper", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&record_item, &settings_item, &quit_item])?;

    let _tray = TrayIconBuilder::with_id("main")
        .icon(app.default_window_icon().unwrap().clone())
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| {
            handle_menu_event(app, event.id.as_ref());
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                // Left click shows the menu (handled by menu_on_left_click)
                let _ = tray;
            }
        })
        .build(app)?;

    Ok(())
}

fn handle_menu_event(app: &AppHandle, id: &str) {
    match id {
        "record" => {
            let state = app.state::<SharedAppState>();

            log::info!("Menu: Toggle recording");

            match crate::state::toggle_recording_with_app(app, &state) {
                Ok(snapshot) => {
                    log::info!("Recording state: {}", snapshot.is_recording);
                }
                Err(e) => {
                    log::error!("Failed to toggle recording: {}", e);
                }
            }
        }
        "settings" => {
            log::info!("Opening settings window");
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }
        "quit" => {
            log::info!("Quitting application");
            app.exit(0);
        }
        _ => {}
    }
}

#[allow(dead_code)]
pub fn update_tray_menu_text(app: &AppHandle, is_recording: bool) {
    // Update the menu item text based on recording state
    if let Some(tray) = app.tray_by_id("main") {
        let text = if is_recording {
            "Stop Recording"
        } else {
            "Start Recording"
        };
        // Note: In Tauri 2.x, updating menu items requires rebuilding the menu
        // This is a simplified version; full implementation would rebuild the menu
        let _ = (tray, text);
    }
}
