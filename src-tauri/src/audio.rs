use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, StreamConfig};
use rubato::{FftFixedIn, Resampler};
use std::sync::{Arc, Mutex};
use thiserror::Error;

const WHISPER_SAMPLE_RATE: u32 = 16000;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No input device available")]
    NoInputDevice,
    #[error("Failed to get default input config: {0}")]
    DefaultConfigError(String),
    #[error("Resampling error: {0}")]
    ResampleError(String),
    #[error("Not recording")]
    NotRecording,
}

/// Audio buffer that can be shared across threads
pub struct AudioBuffer {
    samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    pub fn take(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }
}

/// Thread-safe audio buffer handle
pub type SharedBuffer = Arc<Mutex<AudioBuffer>>;

/// Creates a new shared buffer
pub fn create_shared_buffer() -> SharedBuffer {
    Arc::new(Mutex::new(AudioBuffer::new()))
}

/// Audio recorder configuration
pub struct AudioConfig {
    pub device: Device,
    pub stream_config: StreamConfig,
}

impl AudioConfig {
    pub fn new() -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;

        log::info!("Using input device: {}", device.name().unwrap_or_default());

        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::DefaultConfigError(e.to_string()))?;

        log::info!(
            "Default input config: {:?}, sample rate: {}",
            supported_config.sample_format(),
            supported_config.sample_rate().0
        );

        let stream_config = StreamConfig {
            channels: 1, // Mono
            sample_rate: supported_config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        Ok(Self {
            device,
            stream_config,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.stream_config.sample_rate.0
    }
}

/// Stop recording and get the audio data (resampled to 16kHz)
pub fn stop_recording(
    buffer: &SharedBuffer,
    input_sample_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    log::info!("Stopping recording...");

    let audio = {
        let mut buf = buffer.lock().unwrap();
        buf.take()
    };

    if audio.is_empty() {
        return Err(AudioError::NotRecording);
    }

    log::info!("Recorded {} samples", audio.len());

    // Resample to 16kHz if needed
    if input_sample_rate != WHISPER_SAMPLE_RATE {
        log::info!(
            "Resampling from {}Hz to {}Hz",
            input_sample_rate,
            WHISPER_SAMPLE_RATE
        );
        resample_audio(&audio, input_sample_rate, WHISPER_SAMPLE_RATE)
    } else {
        Ok(audio)
    }
}

/// Silence detection threshold (RMS value)
const SILENCE_THRESHOLD: f32 = 0.01;
/// Minimum silence duration in milliseconds to be considered a split point
const MIN_SILENCE_DURATION_MS: u32 = 300;

/// Calculate RMS (root mean square) of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Find silence boundaries in audio for chunking
/// Returns indices where the audio should be split (at the center of each silence gap)
/// Splits at every silence >= MIN_SILENCE_DURATION_MS (300ms)
pub fn find_silence_boundaries(audio: &[f32], sample_rate: u32) -> Vec<usize> {
    let min_silence_samples = (sample_rate * MIN_SILENCE_DURATION_MS / 1000) as usize;
    let window_size = (sample_rate / 100) as usize; // 10ms windows for RMS calculation

    let mut boundaries = Vec::new();
    let mut silence_start: Option<usize> = None;

    let mut pos = 0;
    while pos + window_size <= audio.len() {
        let rms = calculate_rms(&audio[pos..pos + window_size]);

        if rms < SILENCE_THRESHOLD {
            // In silence
            if silence_start.is_none() {
                silence_start = Some(pos);
            }
        } else {
            // Not in silence - check if we just exited a long enough silence
            if let Some(start) = silence_start {
                let silence_duration = pos - start;

                // Split at every silence >= 300ms
                if silence_duration >= min_silence_samples {
                    let split_point = start + silence_duration / 2; // Split at middle of silence
                    boundaries.push(split_point);
                }
            }
            silence_start = None;
        }

        pos += window_size;
    }

    log::info!(
        "Found {} silence boundaries in {} samples ({:.1}s audio)",
        boundaries.len(),
        audio.len(),
        audio.len() as f32 / sample_rate as f32
    );

    boundaries
}

/// Split audio at the given silence boundaries
/// If no boundaries, returns the entire audio as a single chunk
pub fn split_at_silences(audio: &[f32], boundaries: &[usize]) -> Vec<Vec<f32>> {
    if boundaries.is_empty() {
        // No silence boundaries found - return entire audio as single chunk
        return vec![audio.to_vec()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    for &boundary in boundaries {
        if boundary > start && boundary < audio.len() {
            chunks.push(audio[start..boundary].to_vec());
            start = boundary;
        }
    }

    // Add the final chunk
    if start < audio.len() {
        chunks.push(audio[start..].to_vec());
    }

    log::info!(
        "Split audio into {} chunks: {:?}",
        chunks.len(),
        chunks.iter().map(|c| format!("{:.1}s", c.len() as f32 / 16000.0)).collect::<Vec<_>>()
    );

    chunks
}

fn resample_audio(
    audio: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    let ratio = to_rate as f64 / from_rate as f64;
    let chunk_size = 1024;

    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        2,
        1,
    )
    .map_err(|e| AudioError::ResampleError(e.to_string()))?;

    let mut output = Vec::new();
    let mut input_frames_next = resampler.input_frames_next();

    let mut pos = 0;
    while pos < audio.len() {
        let end = (pos + input_frames_next).min(audio.len());
        let mut chunk: Vec<f32> = audio[pos..end].to_vec();

        // Pad with zeros if needed
        if chunk.len() < input_frames_next {
            chunk.resize(input_frames_next, 0.0);
        }

        let input = vec![chunk];
        let resampled = resampler
            .process(&input, None)
            .map_err(|e| AudioError::ResampleError(e.to_string()))?;

        if !resampled[0].is_empty() {
            output.extend_from_slice(&resampled[0]);
        }

        pos = end;
        input_frames_next = resampler.input_frames_next();
    }

    // Expected output length
    let expected_len = (audio.len() as f64 * ratio) as usize;
    output.truncate(expected_len);

    log::info!(
        "Resampled {} samples to {} samples",
        audio.len(),
        output.len()
    );

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_ratio() {
        let input: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.001).sin()).collect();
        let output = resample_audio(&input, 48000, 16000).unwrap();

        // Should be approximately 1/3 of the input length
        let expected_len = input.len() / 3;
        let tolerance = expected_len / 10; // 10% tolerance
        assert!(
            (output.len() as i64 - expected_len as i64).abs() < tolerance as i64,
            "Expected ~{} samples, got {}",
            expected_len,
            output.len()
        );
    }

    #[test]
    fn test_calculate_rms() {
        // Silence should have RMS near 0
        let silence: Vec<f32> = vec![0.0; 100];
        assert!(calculate_rms(&silence) < 0.001);

        // Loud signal should have higher RMS
        let loud: Vec<f32> = vec![0.5; 100];
        assert!(calculate_rms(&loud) > 0.4);
    }

    #[test]
    fn test_find_silence_in_audio() {
        let sample_rate = 16000;
        // Create audio: 5s speech, 0.5s silence, 5s speech, 0.5s silence, 5s speech
        let mut audio = Vec::new();

        // 5 seconds of "speech" (non-silent signal)
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence (should trigger split - >= 300ms)
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 5 seconds of "speech"
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 5 seconds of "speech"
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find 2 boundaries (at each 500ms silence)
        assert_eq!(boundaries.len(), 2, "Expected 2 boundaries, got {:?}", boundaries);
    }

    #[test]
    fn test_split_at_silences() {
        let audio: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let boundaries = vec![30, 70];

        let chunks = split_at_silences(&audio, &boundaries);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 30);
        assert_eq!(chunks[1].len(), 40);
        assert_eq!(chunks[2].len(), 30);
    }

    #[test]
    fn test_no_silence_returns_single_chunk() {
        let sample_rate = 16000;
        // Create 60 seconds of continuous speech with no silence
        let audio: Vec<f32> = (0..(60 * sample_rate))
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let boundaries = find_silence_boundaries(&audio, sample_rate);
        let chunks = split_at_silences(&audio, &boundaries);

        // No silence found - should return entire audio as single chunk (no hard chunking)
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), audio.len());
    }

    #[test]
    fn test_short_audio_with_silence_is_chunked() {
        let sample_rate = 16000;
        // Create 10 seconds of audio with a silence in the middle
        // Should chunk at the silence regardless of total duration
        let mut audio = Vec::new();

        // 5 seconds of speech
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence (>= 300ms threshold)
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 4.5 seconds of speech
        for i in 0..((4.5 * sample_rate as f32) as usize) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);
        let chunks = split_at_silences(&audio, &boundaries);

        // Should find 1 boundary and create 2 chunks
        assert_eq!(boundaries.len(), 1, "Expected 1 boundary, got {:?}", boundaries);
        assert_eq!(chunks.len(), 2, "Expected 2 chunks, got {}", chunks.len());
    }

    #[test]
    fn test_short_silence_not_split() {
        let sample_rate = 16000;
        // Create audio with a short silence (< 300ms)
        let mut audio = Vec::new();

        // 5 seconds of speech
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.2 seconds of silence (< 300ms threshold)
        for _ in 0..((0.2 * sample_rate as f32) as usize) {
            audio.push(0.0);
        }

        // 5 seconds of speech
        for i in 0..(5 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find no boundaries - silence too short
        assert!(boundaries.is_empty(), "Short silence should not trigger split");
    }
}
