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
/// Target maximum chunk duration in seconds (soft limit, splits at next silence after this)
const TARGET_CHUNK_DURATION_S: u32 = 30;
/// Hard maximum chunk duration in seconds (absolute limit when no silence found)
const MAX_CHUNK_DURATION_S: u32 = 30;

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
pub fn find_silence_boundaries(audio: &[f32], sample_rate: u32) -> Vec<usize> {
    let min_silence_samples = (sample_rate * MIN_SILENCE_DURATION_MS / 1000) as usize;
    let target_chunk_samples = (sample_rate * TARGET_CHUNK_DURATION_S) as usize;
    let window_size = (sample_rate / 100) as usize; // 10ms windows for RMS calculation

    let mut boundaries = Vec::new();
    let mut silence_start: Option<usize> = None;
    let mut last_split = 0;

    let mut pos = 0;
    while pos + window_size <= audio.len() {
        let rms = calculate_rms(&audio[pos..pos + window_size]);

        if rms < SILENCE_THRESHOLD {
            // In silence
            if silence_start.is_none() {
                silence_start = Some(pos);
            }
        } else {
            // Not in silence
            if let Some(start) = silence_start {
                let silence_duration = pos - start;
                let distance_from_last_split = start - last_split;

                // Only split if:
                // 1. Silence is long enough (>300ms)
                // 2. We've accumulated enough audio since last split (>30s target)
                if silence_duration >= min_silence_samples
                    && distance_from_last_split >= target_chunk_samples {
                    let split_point = start + silence_duration / 2; // Split at middle of silence
                    boundaries.push(split_point);
                    last_split = split_point;
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

/// Split audio into fixed-size chunks (hard fallback when no silence found)
fn split_into_fixed_chunks(audio: &[f32], sample_rate: u32) -> Vec<Vec<f32>> {
    let max_chunk_samples = (sample_rate * MAX_CHUNK_DURATION_S) as usize;
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < audio.len() {
        let end = (start + max_chunk_samples).min(audio.len());
        chunks.push(audio[start..end].to_vec());
        start = end;
    }

    log::info!(
        "Hard split audio into {} chunks: {:?}",
        chunks.len(),
        chunks.iter().map(|c| format!("{:.1}s", c.len() as f32 / sample_rate as f32)).collect::<Vec<_>>()
    );

    chunks
}

/// Split audio at the given boundaries, with hard chunking fallback for long audio
pub fn split_at_silences(audio: &[f32], boundaries: &[usize]) -> Vec<Vec<f32>> {
    let max_chunk_samples = (WHISPER_SAMPLE_RATE * MAX_CHUNK_DURATION_S) as usize;

    if boundaries.is_empty() {
        // No silence boundaries found - use hard chunking if audio is too long
        if audio.len() > max_chunk_samples {
            log::info!("No silence boundaries found, using hard chunking fallback");
            return split_into_fixed_chunks(audio, WHISPER_SAMPLE_RATE);
        }
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

    // Check if any chunks are too long and need hard splitting
    let mut final_chunks = Vec::new();
    for chunk in chunks {
        if chunk.len() > max_chunk_samples {
            log::info!("Chunk too long ({:.1}s), applying hard split", chunk.len() as f32 / WHISPER_SAMPLE_RATE as f32);
            final_chunks.extend(split_into_fixed_chunks(&chunk, WHISPER_SAMPLE_RATE));
        } else {
            final_chunks.push(chunk);
        }
    }

    log::info!(
        "Split audio into {} chunks: {:?}",
        final_chunks.len(),
        final_chunks.iter().map(|c| format!("{:.1}s", c.len() as f32 / 16000.0)).collect::<Vec<_>>()
    );

    final_chunks
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
        // Create 90 seconds of audio: 35s speech, 0.5s silence, 35s speech, 0.5s silence, 18s speech
        let mut audio = Vec::new();

        // 35 seconds of "speech" (non-silent signal)
        for i in 0..(35 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence (should trigger split after 30s+ of speech)
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 35 seconds of "speech"
        for i in 0..(35 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 18 seconds of "speech"
        for i in 0..(18 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find 2 boundaries (after first 35s silence, after second 35s silence)
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
    fn test_short_no_silence_returns_single_chunk() {
        let sample_rate = 16000;
        // Create 25 seconds of continuous speech with no silence (under 30s limit)
        let audio: Vec<f32> = (0..(25 * sample_rate))
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let boundaries = find_silence_boundaries(&audio, sample_rate);
        let chunks = split_at_silences(&audio, &boundaries);

        // No silence found but under 30s, should return single chunk
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), audio.len());
    }

    #[test]
    fn test_hard_chunking_fallback_when_no_silence() {
        let sample_rate = 16000;
        // Create 90 seconds of continuous speech with no silence
        let audio: Vec<f32> = (0..(90 * sample_rate))
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let boundaries = find_silence_boundaries(&audio, sample_rate);
        let chunks = split_at_silences(&audio, &boundaries);

        // No silence found but >30s, should use hard chunking fallback
        // 90s audio should be split into 3 chunks of ~30s each
        assert_eq!(chunks.len(), 3, "Expected 3 chunks for 90s audio, got {}", chunks.len());

        // Each chunk should be approximately 30 seconds (480000 samples at 16kHz)
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_duration_s = chunk.len() as f32 / sample_rate as f32;
            assert!(
                chunk_duration_s <= 30.0,
                "Chunk {} is {:.1}s, should be <= 30s",
                i,
                chunk_duration_s
            );
        }
    }

    #[test]
    fn test_short_audio_not_chunked() {
        let sample_rate = 16000;
        // Create 20 seconds of audio with a silence in the middle
        // Should NOT chunk because total audio is less than 30s target
        let mut audio = Vec::new();

        // 10 seconds of speech
        for i in 0..(10 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence
        for _ in 0..(sample_rate / 2) {
            audio.push(0.0);
        }

        // 9.5 seconds of speech
        for i in 0..((9.5 * sample_rate as f32) as usize) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find no boundaries because we haven't hit 30s yet
        assert!(boundaries.is_empty(), "Short audio should not be chunked");
    }
}
