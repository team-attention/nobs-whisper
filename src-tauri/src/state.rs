use crate::audio::{self, AudioConfig, SharedBuffer};
use crate::config::AppConfig;
use crate::indicator;
use crate::input;
use crate::whisper::WhisperEngine;
use cpal::traits::{DeviceTrait, HostTrait};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::thread::JoinHandle;
use tauri::{AppHandle, Emitter, State};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppStateSnapshot {
    pub is_recording: bool,
    pub is_transcribing: bool,
    pub last_transcription: Option<String>,
    pub error: Option<String>,
    pub selected_model: Option<String>,
}

/// The recording state, kept separate since cpal::Stream is not Send
pub struct RecordingState {
    pub audio_config: Option<AudioConfig>,
    pub audio_buffer: Option<SharedBuffer>,
    /// Channel to signal the recording thread to stop
    pub stop_sender: Option<Sender<()>>,
    /// Handle to the recording thread for cleanup
    pub recording_thread: Option<JoinHandle<()>>,
    // Note: The cpal::Stream is intentionally NOT stored here
    // It will be managed by a dedicated audio thread
}

impl RecordingState {
    pub fn new() -> Self {
        Self {
            audio_config: None,
            audio_buffer: None,
            stop_sender: None,
            recording_thread: None,
        }
    }

    /// Clean up any previous recording thread (non-blocking)
    pub fn cleanup_previous_thread(&mut self) {
        if let Some(handle) = self.recording_thread.take() {
            // Check if thread is finished without blocking
            if handle.is_finished() {
                log::info!("Previous recording thread already finished");
                let _ = handle.join();
            } else {
                // Thread still running - spawn a cleanup thread to join it
                log::warn!("Previous recording thread still running, spawning cleanup thread");
                std::thread::spawn(move || {
                    let _ = handle.join();
                    log::info!("Previous recording thread cleaned up");
                });
            }
        }
    }
}

/// The main app state (Send + Sync safe)
pub struct AppState {
    pub config: AppConfig,
    pub is_recording: bool,
    pub is_transcribing: bool,
    pub last_transcription: Option<String>,
    pub error: Option<String>,
    pub whisper_engine: Option<Arc<WhisperEngine>>,
    pub recording_state: RecordingState,
    /// Currently registered shortcut (for tracking to unregister on change)
    pub active_shortcut: Option<String>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            is_recording: false,
            is_transcribing: false,
            last_transcription: None,
            error: None,
            whisper_engine: None,
            recording_state: RecordingState::new(),
            active_shortcut: None,
        }
    }

    pub fn snapshot(&self) -> AppStateSnapshot {
        AppStateSnapshot {
            is_recording: self.is_recording,
            is_transcribing: self.is_transcribing,
            last_transcription: self.last_transcription.clone(),
            error: self.error.clone(),
            selected_model: self.config.selected_model.clone(),
        }
    }
}

// Use std::sync::Mutex instead of tokio::sync::Mutex for the state
// because we need it to be Sync for Tauri
pub type SharedAppState = Arc<StdMutex<AppState>>;

#[tauri::command]
pub fn get_app_state(
    state: State<'_, SharedAppState>,
) -> Result<AppStateSnapshot, String> {
    let state = state.lock().map_err(|e| e.to_string())?;
    Ok(state.snapshot())
}

#[tauri::command]
pub fn toggle_recording(
    state: State<'_, SharedAppState>,
) -> Result<AppStateSnapshot, String> {
    let mut state_guard = state.lock().map_err(|e| e.to_string())?;

    if state_guard.is_recording {
        // Stop recording
        log::info!("Stopping recording...");
        state_guard.is_recording = false;

        // Signal recording thread to stop
        if let Some(sender) = state_guard.recording_state.stop_sender.take() {
            let _ = sender.send(());
        }

        // Note: Don't join here - let the thread cleanup happen on next recording start
        // This prevents blocking while holding the state mutex

        // Get audio data
        let audio_data = if let Some(ref buffer) = state_guard.recording_state.audio_buffer {
            let sample_rate = state_guard
                .recording_state
                .audio_config
                .as_ref()
                .map(|c| c.sample_rate())
                .unwrap_or(48000);

            match audio::stop_recording(buffer, sample_rate) {
                Ok(data) => {
                    log::info!("Got {} audio samples", data.len());
                    Some(data)
                }
                Err(e) => {
                    log::error!("Failed to stop recording: {}", e);
                    state_guard.error = Some(format!("Failed to stop recording: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Transcribe if we have audio
        if let Some(audio) = audio_data {
            if audio.len() > 1600 {
                // At least 0.1 second at 16kHz
                state_guard.is_transcribing = true;

                // Clone whisper engine Arc and get language/vocabulary before releasing lock
                let whisper_opt = state_guard.whisper_engine.clone();
                let language_owned = if state_guard.config.language == "auto" {
                    None
                } else {
                    Some(state_guard.config.language.clone())
                };
                let vocabulary_owned = if state_guard.config.custom_vocabulary.is_empty() {
                    None
                } else {
                    Some(state_guard.config.custom_vocabulary.clone())
                };

                // Release lock before transcription
                drop(state_guard);

                if let Some(whisper) = whisper_opt {
                    let language_ref = language_owned.as_deref();
                    let vocabulary_ref = vocabulary_owned.as_deref();

                    match whisper.transcribe(&audio, language_ref, vocabulary_ref) {
                        Ok(text) => {
                            log::info!("Transcription: {}", text);

                            // Re-acquire lock to update state
                            let mut state_guard = state.lock().map_err(|e| e.to_string())?;
                            state_guard.last_transcription = Some(text.clone());
                            state_guard.is_transcribing = false;
                            drop(state_guard);

                            // Type the text or copy to clipboard
                            match type_transcription(&text) {
                                Ok(true) => log::info!("Text typed successfully"),
                                Ok(false) => log::info!("Text copied to clipboard (no focused input)"),
                                Err(e) => log::error!("Failed to type/copy text: {}", e),
                            }

                            // Re-acquire lock for snapshot
                            let state_guard = state.lock().map_err(|e| e.to_string())?;
                            return Ok(state_guard.snapshot());
                        }
                        Err(e) => {
                            log::error!("Transcription failed: {}", e);
                            let mut state_guard = state.lock().map_err(|e| e.to_string())?;
                            state_guard.error = Some(format!("Transcription failed: {}", e));
                            state_guard.is_transcribing = false;
                            return Ok(state_guard.snapshot());
                        }
                    }
                } else {
                    let mut state_guard = state.lock().map_err(|e| e.to_string())?;
                    state_guard.error = Some("No model loaded".to_string());
                    state_guard.is_transcribing = false;
                    return Ok(state_guard.snapshot());
                }
            }
        }
        // Audio too short or no audio - re-acquire lock for final state
        let state_guard = state.lock().map_err(|e| e.to_string())?;
        return Ok(state_guard.snapshot());
    } else {
        // Start recording
        log::info!("Starting recording...");
        state_guard.error = None;

        // Cleanup any previous recording thread first
        state_guard.recording_state.cleanup_previous_thread();

        // Initialize audio config if needed
        if state_guard.recording_state.audio_config.is_none() {
            match AudioConfig::new() {
                Ok(config) => {
                    state_guard.recording_state.audio_config = Some(config);
                }
                Err(e) => {
                    log::error!("Failed to create audio config: {}", e);
                    state_guard.error = Some(format!("Failed to create audio config: {}", e));
                    return Ok(state_guard.snapshot());
                }
            }
        }

        // Create buffer and start recording
        let buffer = audio::create_shared_buffer();
        state_guard.recording_state.audio_buffer = Some(buffer.clone());

        // Create stop channel
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        state_guard.recording_state.stop_sender = Some(stop_tx);

        // Get max duration from config (0 = unlimited, capped at 600s)
        let max_duration = state_guard.config.max_recording_duration;
        let effective_duration = if max_duration == 0 { 600 } else { max_duration.min(600) };

        if let Some(ref config) = state_guard.recording_state.audio_config {
            // Start recording on a dedicated thread since cpal::Stream is not Send
            let device_name = config.device.name().unwrap_or_default();
            let _sample_rate = config.sample_rate();
            let stream_config = config.stream_config.clone();

            // Clone the device - we need to create the stream on the recording thread
            match cpal::default_host().default_input_device() {
                Some(device) => {
                    state_guard.is_recording = true;
                    log::info!("Recording started on device: {}", device_name);

                    // Spawn recording thread and store handle for cleanup
                    let handle = std::thread::spawn(move || {
                        let buffer_clone = buffer.clone();
                        let channels = stream_config.channels as usize;

                        // Build and run the stream on this thread
                        let stream = device.build_input_stream(
                            &stream_config,
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                let mut buf = buffer_clone.lock().unwrap();
                                // Convert to mono if stereo
                                if channels > 1 {
                                    for chunk in data.chunks(channels) {
                                        let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                                        buf.push_samples(&[mono]);
                                    }
                                } else {
                                    buf.push_samples(data);
                                }
                            },
                            |err| {
                                log::error!("Audio input error: {}", err);
                            },
                            None,
                        );

                        match stream {
                            Ok(s) => {
                                use cpal::traits::StreamTrait;
                                if let Err(e) = s.play() {
                                    log::error!("Failed to play stream: {}", e);
                                    return;
                                }
                                // Wait for stop signal or timeout
                                let timeout = std::time::Duration::from_secs(effective_duration);
                                match stop_rx.recv_timeout(timeout) {
                                    Ok(()) => log::info!("Recording stopped by user"),
                                    Err(mpsc::RecvTimeoutError::Timeout) => {
                                        log::info!("Recording stopped by timeout ({}s)", effective_duration)
                                    }
                                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                                        log::info!("Recording channel disconnected")
                                    }
                                }
                                // Stream is dropped here, releasing the audio device
                                log::info!("Recording thread exiting, releasing audio resources");
                            }
                            Err(e) => {
                                log::error!("Failed to build stream: {}", e);
                            }
                        }
                    });
                    state_guard.recording_state.recording_thread = Some(handle);
                }
                None => {
                    log::error!("No input device available");
                    state_guard.error = Some("No input device available".to_string());
                }
            }
        }
    }

    Ok(state_guard.snapshot())
}

fn type_transcription(text: &str) -> Result<bool, String> {
    input::type_or_copy(text).map_err(|e| e.to_string())
}

/// Emit the state-changed event and update indicator window
fn emit_state_change(app: &AppHandle, snapshot: &AppStateSnapshot) {
    log::info!(
        "emit_state_change: is_recording={}, is_transcribing={}",
        snapshot.is_recording,
        snapshot.is_transcribing
    );

    // Emit event for frontend
    if let Err(e) = app.emit("state-changed", snapshot) {
        log::error!("Failed to emit state-changed event: {}", e);
    }

    // Show/hide indicator based on state
    if snapshot.is_recording {
        // Set status to recording and show
        let _ = indicator::set_indicator_status(app, "recording");
        if let Err(e) = indicator::show_indicator(app) {
            log::error!("Failed to show indicator: {}", e);
        }
    } else if snapshot.is_transcribing {
        // Set status to processing and show
        let _ = indicator::set_indicator_status(app, "processing");
        if let Err(e) = indicator::show_indicator(app) {
            log::error!("Failed to show indicator: {}", e);
        }
    } else {
        log::info!("Hiding indicator");
        if let Err(e) = indicator::hide_indicator(app) {
            log::error!("Failed to hide indicator: {}", e);
        }
    }
}

/// Toggle recording with app handle for event emission
pub fn toggle_recording_with_app(
    app: &AppHandle,
    state: &SharedAppState,
) -> Result<AppStateSnapshot, String> {
    let mut state_guard = state.lock().map_err(|e| e.to_string())?;

    if state_guard.is_recording {
        // Stop recording
        log::info!("Stopping recording...");
        state_guard.is_recording = false;

        // Signal recording thread to stop
        if let Some(sender) = state_guard.recording_state.stop_sender.take() {
            let _ = sender.send(());
        }

        // Note: Don't join here - let the thread cleanup happen on next recording start
        // This prevents blocking while holding the state mutex

        // Get buffer and sample rate, then release lock BEFORE resampling
        let buffer_opt = state_guard.recording_state.audio_buffer.clone();
        let sample_rate = state_guard
            .recording_state
            .audio_config
            .as_ref()
            .map(|c| c.sample_rate())
            .unwrap_or(48000);

        // Clone whisper engine Arc and get language/vocabulary before releasing lock
        let whisper_opt = state_guard.whisper_engine.clone();
        let language_owned = if state_guard.config.language == "auto" {
            None
        } else {
            Some(state_guard.config.language.clone())
        };
        let vocabulary_owned = if state_guard.config.custom_vocabulary.is_empty() {
            None
        } else {
            Some(state_guard.config.custom_vocabulary.clone())
        };

        // Set transcribing state and release lock BEFORE any heavy processing
        state_guard.is_transcribing = true;
        let snapshot = state_guard.snapshot();
        drop(state_guard);

        // Emit state change - frontend can poll freely now
        emit_state_change(app, &snapshot);

        // Spawn a thread for audio processing and transcription
        // This prevents blocking the main thread so Tauri can handle get_app_state
        let state_clone = state.clone();
        let app_clone = app.clone();

        std::thread::spawn(move || {
            // Do audio processing in this thread
            let audio_data = if let Some(ref buffer) = buffer_opt {
                match audio::stop_recording(buffer, sample_rate) {
                    Ok(data) => {
                        log::info!("Got {} audio samples", data.len());
                        Some(data)
                    }
                    Err(e) => {
                        log::error!("Failed to stop recording: {}", e);
                        if let Ok(mut state_guard) = state_clone.lock() {
                            state_guard.error = Some(format!("Failed to stop recording: {}", e));
                            state_guard.is_transcribing = false;
                            let snapshot = state_guard.snapshot();
                            emit_state_change(&app_clone, &snapshot);
                        }
                        return;
                    }
                }
            } else {
                None
            };

            // Transcribe if we have audio
            if let Some(audio) = audio_data {
                if audio.len() > 1600 {
                    // At least 0.1 second at 16kHz
                    if let Some(whisper) = whisper_opt {
                        let language_ref = language_owned.as_deref();
                        let vocabulary_ref = vocabulary_owned.as_deref();

                        // Use chunked transcription for long audio (>30s = 480,000 samples at 16kHz)
                        let transcription_result = if audio.len() > 480_000 {
                            let boundaries = audio::find_silence_boundaries(&audio, 16000);
                            let chunks = audio::split_at_silences(&audio, &boundaries);
                            if chunks.len() > 1 {
                                log::info!("Using chunked transcription: {} chunks", chunks.len());
                                whisper.transcribe_chunked(chunks, language_ref, vocabulary_ref)
                            } else {
                                whisper.transcribe(&audio, language_ref, vocabulary_ref)
                            }
                        } else {
                            whisper.transcribe(&audio, language_ref, vocabulary_ref)
                        };

                        match transcription_result {
                            Ok(text) => {
                                log::info!("Transcription: {}", text);

                                // Update state
                                if let Ok(mut state_guard) = state_clone.lock() {
                                    state_guard.last_transcription = Some(text.clone());
                                    state_guard.is_transcribing = false;
                                    let final_snapshot = state_guard.snapshot();
                                    drop(state_guard);

                                    // Type the text or copy to clipboard - must run on main thread
                                    let text_to_type = text.clone();
                                    let app_for_typing = app_clone.clone();
                                    let snapshot_for_typing = final_snapshot.clone();
                                    let _ = app_clone.run_on_main_thread(move || {
                                        match type_transcription(&text_to_type) {
                                            Ok(true) => {
                                                log::info!("Text typed successfully");
                                                emit_state_change(&app_for_typing, &snapshot_for_typing);
                                            }
                                            Ok(false) => {
                                                log::info!("Text copied to clipboard (no focused input)");
                                                let _ = indicator::set_indicator_status(&app_for_typing, "copied");
                                                // Emit state change on a separate thread after delay
                                                // to avoid blocking the main thread
                                                let app_delayed = app_for_typing.clone();
                                                let snapshot_delayed = snapshot_for_typing.clone();
                                                std::thread::spawn(move || {
                                                    std::thread::sleep(std::time::Duration::from_millis(1000));
                                                    let app_for_emit = app_delayed.clone();
                                                    let _ = app_delayed.run_on_main_thread(move || {
                                                        emit_state_change(&app_for_emit, &snapshot_delayed);
                                                    });
                                                });
                                            }
                                            Err(e) => {
                                                log::error!("Failed to type/copy text: {}", e);
                                                emit_state_change(&app_for_typing, &snapshot_for_typing);
                                            }
                                        }
                                    });
                                }
                            }
                            Err(e) => {
                                log::error!("Transcription failed: {}", e);
                                if let Ok(mut state_guard) = state_clone.lock() {
                                    state_guard.error = Some(format!("Transcription failed: {}", e));
                                    state_guard.is_transcribing = false;
                                    let snapshot = state_guard.snapshot();
                                    emit_state_change(&app_clone, &snapshot);
                                }
                            }
                        }
                        return;
                    } else {
                        // No model loaded
                        if let Ok(mut state_guard) = state_clone.lock() {
                            state_guard.error = Some("No model loaded".to_string());
                            state_guard.is_transcribing = false;
                            let snapshot = state_guard.snapshot();
                            emit_state_change(&app_clone, &snapshot);
                        }
                        return;
                    }
                }
            }

            // Audio too short or no audio
            if let Ok(mut state_guard) = state_clone.lock() {
                state_guard.is_transcribing = false;
                let snapshot = state_guard.snapshot();
                emit_state_change(&app_clone, &snapshot);
            }
        });

        // Return immediately with current snapshot
        return Ok(snapshot);
    } else {
        // Start recording
        log::info!("Starting recording...");
        state_guard.error = None;

        // Cleanup any previous recording thread first
        state_guard.recording_state.cleanup_previous_thread();

        // Initialize audio config if needed
        if state_guard.recording_state.audio_config.is_none() {
            match AudioConfig::new() {
                Ok(config) => {
                    state_guard.recording_state.audio_config = Some(config);
                }
                Err(e) => {
                    log::error!("Failed to create audio config: {}", e);
                    state_guard.error = Some(format!("Failed to create audio config: {}", e));
                    let snapshot = state_guard.snapshot();
                    emit_state_change(app, &snapshot);
                    return Ok(snapshot);
                }
            }
        }

        // Create buffer and start recording
        let buffer = audio::create_shared_buffer();
        state_guard.recording_state.audio_buffer = Some(buffer.clone());

        // Create stop channel
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        state_guard.recording_state.stop_sender = Some(stop_tx);

        // Get max duration from config (0 = unlimited, capped at 600s)
        let max_duration = state_guard.config.max_recording_duration;
        let effective_duration = if max_duration == 0 { 600 } else { max_duration.min(600) };

        if let Some(ref config) = state_guard.recording_state.audio_config {
            // Start recording on a dedicated thread since cpal::Stream is not Send
            let device_name = config.device.name().unwrap_or_default();
            let _sample_rate = config.sample_rate();
            let stream_config = config.stream_config.clone();

            // Clone the device - we need to create the stream on the recording thread
            match cpal::default_host().default_input_device() {
                Some(device) => {
                    state_guard.is_recording = true;
                    log::info!("Recording started on device: {}", device_name);

                    // Spawn recording thread and store handle for cleanup
                    let handle = std::thread::spawn(move || {
                        let buffer_clone = buffer.clone();
                        let channels = stream_config.channels as usize;

                        // Build and run the stream on this thread
                        let stream = device.build_input_stream(
                            &stream_config,
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                let mut buf = buffer_clone.lock().unwrap();
                                // Convert to mono if stereo
                                if channels > 1 {
                                    for chunk in data.chunks(channels) {
                                        let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                                        buf.push_samples(&[mono]);
                                    }
                                } else {
                                    buf.push_samples(data);
                                }
                            },
                            |err| {
                                log::error!("Audio input error: {}", err);
                            },
                            None,
                        );

                        match stream {
                            Ok(s) => {
                                use cpal::traits::StreamTrait;
                                if let Err(e) = s.play() {
                                    log::error!("Failed to play stream: {}", e);
                                    return;
                                }
                                // Wait for stop signal or timeout
                                let timeout = std::time::Duration::from_secs(effective_duration);
                                match stop_rx.recv_timeout(timeout) {
                                    Ok(()) => log::info!("Recording stopped by user"),
                                    Err(mpsc::RecvTimeoutError::Timeout) => {
                                        log::info!("Recording stopped by timeout ({}s)", effective_duration)
                                    }
                                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                                        log::info!("Recording channel disconnected")
                                    }
                                }
                                // Stream is dropped here, releasing the audio device
                                log::info!("Recording thread exiting, releasing audio resources");
                            }
                            Err(e) => {
                                log::error!("Failed to build stream: {}", e);
                            }
                        }
                    });
                    state_guard.recording_state.recording_thread = Some(handle);
                }
                None => {
                    log::error!("No input device available");
                    state_guard.error = Some("No input device available".to_string());
                }
            }
        }
    }

    let snapshot = state_guard.snapshot();
    emit_state_change(app, &snapshot);
    Ok(snapshot)
}
