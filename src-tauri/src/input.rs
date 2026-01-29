use arboard::Clipboard;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InputError {
    #[error("Failed to initialize input system: {0}")]
    InitError(String),
    #[error("Failed to type text: {0}")]
    TypeError(String),
    #[error("Clipboard error: {0}")]
    ClipboardError(String),
    #[error("Accessibility permission required")]
    #[allow(dead_code)]
    NoPermission,
    #[error("No focused input field")]
    NoFocusedInput,
}

pub struct TextInputter;

impl TextInputter {
    pub fn new() -> Result<Self, InputError> {
        Ok(Self)
    }

    pub fn type_text(&mut self, text: &str) -> Result<(), InputError> {
        log::info!("Typing text: {} chars", text.len());

        // Use clipboard paste - more reliable than direct typing
        // Direct typing with enigo can partially succeed then error,
        // causing duplicate input when falling back to paste
        self.paste_from_clipboard(text)
    }

    fn paste_from_clipboard(&mut self, text: &str) -> Result<(), InputError> {
        // Save current clipboard content
        let mut clipboard = Clipboard::new()
            .map_err(|e| InputError::ClipboardError(e.to_string()))?;

        let previous_content = clipboard.get_text().ok();

        // Set new text to clipboard
        clipboard
            .set_text(text)
            .map_err(|e| InputError::ClipboardError(e.to_string()))?;

        // Small delay to let clipboard settle and input method stabilize
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Simulate Cmd+V on macOS using CGEvent (bypasses IME interception)
        #[cfg(target_os = "macos")]
        {
            use core_graphics::event::{CGEvent, CGEventFlags, CGKeyCode};
            use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};

            const KV_ANSI_V: CGKeyCode = 0x09;

            let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState)
                .map_err(|_| InputError::TypeError("Failed to create CGEventSource".to_string()))?;

            // Key down: Cmd+V
            let key_down = CGEvent::new_keyboard_event(source.clone(), KV_ANSI_V, true)
                .map_err(|_| InputError::TypeError("Failed to create keydown event".to_string()))?;
            key_down.set_flags(CGEventFlags::CGEventFlagCommand);
            key_down.post(core_graphics::event::CGEventTapLocation::HID);

            std::thread::sleep(std::time::Duration::from_millis(20));

            // Key up: Cmd+V
            let key_up = CGEvent::new_keyboard_event(source, KV_ANSI_V, false)
                .map_err(|_| InputError::TypeError("Failed to create keyup event".to_string()))?;
            key_up.set_flags(CGEventFlags::CGEventFlagCommand);
            key_up.post(core_graphics::event::CGEventTapLocation::HID);
        }

        // Wait a bit for paste to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Restore previous clipboard content if available
        if let Some(content) = previous_content {
            let _ = clipboard.set_text(&content);
        }

        log::info!("Text pasted via clipboard");
        Ok(())
    }
}

pub fn copy_to_clipboard(text: &str) -> Result<(), InputError> {
    let mut clipboard = Clipboard::new()
        .map_err(|e| InputError::ClipboardError(e.to_string()))?;

    clipboard
        .set_text(text)
        .map_err(|e| InputError::ClipboardError(e.to_string()))?;

    Ok(())
}

/// Check if there's a focused text input field (macOS only)
#[cfg(target_os = "macos")]
pub fn has_focused_input() -> bool {
    use std::process::Command;

    // Use AppleScript to check if there's a focused text field
    // Uses AXFocusedUIElement attribute which is more reliable
    let script = r#"
        tell application "System Events"
            try
                set frontApp to first application process whose frontmost is true
                set focusedUI to value of attribute "AXFocusedUIElement" of frontApp
                if focusedUI is not missing value then
                    set uiRole to value of attribute "AXRole" of focusedUI
                    if uiRole is "AXTextField" or uiRole is "AXTextArea" or uiRole is "AXComboBox" or uiRole is "AXSearchField" or uiRole is "AXWebArea" then
                        return "true:" & uiRole
                    end if
                    return "false:" & uiRole
                end if
            end try
        end tell
        return "false:none"
    "#;

    match Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
    {
        Ok(output) => {
            let result = String::from_utf8_lossy(&output.stdout);
            let result = result.trim();
            log::info!("has_focused_input check: {}", result);
            result.starts_with("true")
        }
        Err(e) => {
            log::error!("has_focused_input AppleScript error: {}", e);
            // If we can't check, assume there's a focused input
            true
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn has_focused_input() -> bool {
    // On other platforms, assume there's always a focused input
    true
}

/// Show a macOS notification
#[cfg(target_os = "macos")]
pub fn show_notification(title: &str, message: &str) {
    use std::process::Command;

    let script = format!(
        r#"display notification "{}" with title "{}""#,
        message.replace('"', "\\\""),
        title.replace('"', "\\\"")
    );

    let _ = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output();
}

#[cfg(not(target_os = "macos"))]
pub fn show_notification(_title: &str, _message: &str) {
    // No-op on other platforms
}

/// Type text or copy to clipboard if no focused input
pub fn type_or_copy(text: &str) -> Result<bool, InputError> {
    // Check if there's a focused input field
    if !has_focused_input() {
        log::info!("No focused input field, copying to clipboard");
        copy_to_clipboard(text)?;
        return Ok(false); // false = copied to clipboard
    }

    // Try to type
    let mut inputter = TextInputter::new()?;
    inputter.type_text(text)?;
    Ok(true) // true = typed successfully
}

#[tauri::command]
pub fn check_accessibility_permission() -> bool {
    // Try to create an Enigo instance to check if we have permission
    let settings = enigo::Settings::default();
    enigo::Enigo::new(&settings).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clipboard() {
        let text = "Test clipboard content";
        let result = copy_to_clipboard(text);
        assert!(result.is_ok());
    }
}
