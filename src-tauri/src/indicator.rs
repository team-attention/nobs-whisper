use tauri::AppHandle;

#[cfg(target_os = "macos")]
use cocoa::base::{id, nil};
#[cfg(target_os = "macos")]
use cocoa::foundation::NSString;
#[cfg(target_os = "macos")]
use objc::{class, msg_send, sel, sel_impl};
#[cfg(target_os = "macos")]
use std::process::{Child, Command};
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use tauri::Manager;

#[cfg(target_os = "macos")]
static HELPER_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

/// Kill any existing indicator helper processes
#[cfg(target_os = "macos")]
fn kill_existing_helpers() {
    // Send terminate notification to any running helpers
    post_notification("com.nobswhisper.indicator.terminate", None);

    // Also kill any old ListenIndicator processes that might conflict
    let _ = Command::new("pkill")
        .args(["-f", "ListenIndicator"])
        .output();

    // Kill our own indicator processes
    let _ = Command::new("pkill")
        .args(["-f", "NobsWhisperIndicator"])
        .output();

    // Give it a moment to terminate
    std::thread::sleep(std::time::Duration::from_millis(100));
}

/// Start the native indicator helper app
#[cfg(target_os = "macos")]
pub fn start_indicator_helper(app: &AppHandle) {
    // Kill any existing helpers first
    kill_existing_helpers();

    let mut helper = HELPER_PROCESS.lock().unwrap();

    // Clear any stale process handle
    if let Some(mut child) = helper.take() {
        let _ = child.kill();
    }

    // Try to find the helper executable
    // First check in the app bundle (for production)
    let helper_path: Option<std::path::PathBuf> = if let Ok(resource_dir) = app.path().resource_dir() {
        // Tauri bundles "../" paths as "_up_/" in resources
        let bundled_path = resource_dir.join("_up_/NobsWhisperIndicator/.build/release/NobsWhisperIndicator");
        if bundled_path.exists() {
            Some(bundled_path)
        } else {
            // Try direct path as fallback
            let direct_path = resource_dir.join("NobsWhisperIndicator");
            if direct_path.exists() {
                Some(direct_path)
            } else {
                None
            }
        }
    } else {
        None
    };

    // Fall back to development path
    let helper_path = helper_path.unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_default()
            .parent()
            .map(|p| p.join("NobsWhisperIndicator/.build/release/NobsWhisperIndicator"))
            .unwrap_or_default()
    });

    log::info!("Starting indicator helper from: {:?}", helper_path);

    match Command::new(&helper_path).spawn() {
        Ok(child) => {
            *helper = Some(child);
            log::info!("Indicator helper started successfully");
        }
        Err(e) => {
            log::error!("Failed to start indicator helper: {}", e);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub fn start_indicator_helper(_app: &AppHandle) {
    // No-op on other platforms
}

/// Stop the indicator helper app
#[cfg(target_os = "macos")]
pub fn stop_indicator_helper() {
    // Send terminate notification
    post_notification("com.nobswhisper.indicator.terminate", None);

    // Also kill the process if it's still running
    let mut helper = HELPER_PROCESS.lock().unwrap();
    if let Some(mut child) = helper.take() {
        let _ = child.kill();
    }
}

#[cfg(not(target_os = "macos"))]
pub fn stop_indicator_helper() {
    // No-op on other platforms
}

/// Post a distributed notification to the helper app
#[cfg(target_os = "macos")]
fn post_notification(name: &str, user_info: Option<&str>) {
    unsafe {
        let center: id = msg_send![class!(NSDistributedNotificationCenter), defaultCenter];
        let notification_name = NSString::alloc(nil).init_str(name);

        let user_info_dict: id = if let Some(status) = user_info {
            let key = NSString::alloc(nil).init_str("status");
            let value = NSString::alloc(nil).init_str(status);
            let keys = vec![key];
            let objects = vec![value];
            cocoa::foundation::NSDictionary::dictionaryWithObjects_forKeys_(
                nil,
                cocoa::foundation::NSArray::arrayWithObjects(nil, &objects),
                cocoa::foundation::NSArray::arrayWithObjects(nil, &keys),
            )
        } else {
            nil
        };

        let null_object: id = nil;
        let _: () = msg_send![center,
            postNotificationName: notification_name
            object: null_object
            userInfo: user_info_dict
            deliverImmediately: true
        ];
    }
}

/// Show the indicator
pub fn show_indicator(_app: &AppHandle) -> Result<(), String> {
    log::info!("show_indicator called");

    #[cfg(target_os = "macos")]
    {
        post_notification("com.nobswhisper.indicator.show", None);
        log::info!("Sent show notification to indicator helper");
    }

    Ok(())
}

/// Hide the indicator
pub fn hide_indicator(_app: &AppHandle) -> Result<(), String> {
    log::info!("hide_indicator called");

    #[cfg(target_os = "macos")]
    {
        post_notification("com.nobswhisper.indicator.hide", None);
        log::info!("Sent hide notification to indicator helper");
    }

    Ok(())
}

/// Set indicator status (recording or processing)
pub fn set_indicator_status(_app: &AppHandle, status: &str) -> Result<(), String> {
    log::info!("set_indicator_status called with: {}", status);

    #[cfg(target_os = "macos")]
    {
        post_notification("com.nobswhisper.indicator.status", Some(status));
        log::info!("Sent status notification to indicator helper");
    }

    Ok(())
}

/// Setup indicator (called from setup hook) - now starts the helper
#[cfg(target_os = "macos")]
pub fn setup_indicator_window(app: &AppHandle) {
    start_indicator_helper(app);
}

#[cfg(not(target_os = "macos"))]
pub fn setup_indicator_window(_app: &AppHandle) {
    // No-op on other platforms
}
