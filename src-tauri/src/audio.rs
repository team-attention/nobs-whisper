use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{Device, StreamConfig};
use rubato::{FftFixedIn, Resampler};
use std::sync::{Arc, Mutex};
use thiserror::Error;

const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Overlap duration in milliseconds for chunk boundaries
/// This prevents word cutting at chunk edges by including some audio from the previous chunk
const CHUNK_OVERLAP_MS: u32 = 200;

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
    /// Position of the last non-silent sample (for real-time silence detection)
    last_speech_pos: usize,
    /// Sample rate for silence detection calculations
    sample_rate: u32,
    /// Adaptive noise floor (background noise level)
    noise_floor: f32,
    /// Number of frames used to calculate noise floor
    noise_floor_frames: usize,
    /// Overlap samples to keep from previous chunk
    overlap_buffer: Vec<f32>,
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self::with_sample_rate(48000) // Default to common input sample rate
    }

    pub fn with_sample_rate(sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            last_speech_pos: 0,
            sample_rate,
            noise_floor: SILENCE_THRESHOLD, // Start with default threshold
            noise_floor_frames: 0,
            overlap_buffer: Vec::new(),
        }
    }

    pub fn push_samples(&mut self, samples: &[f32]) {
        let start_pos = self.samples.len();
        self.samples.extend_from_slice(samples);

        // Check if new samples contain speech (non-silent audio)
        let window_size = (self.sample_rate / 50) as usize; // 20ms windows for more stable RMS
        for (i, chunk) in samples.chunks(window_size).enumerate() {
            let rms = calculate_rms(chunk);

            // Update adaptive noise floor during quiet periods
            // Only update if this looks like background noise (very low energy)
            if rms < self.noise_floor * 0.5 && self.noise_floor_frames < 100 {
                // Exponential moving average for noise floor
                self.noise_floor = self.noise_floor * 0.95 + rms * 0.05;
                self.noise_floor_frames += 1;
            }

            // Use adaptive threshold: 3x noise floor, but at least MIN threshold
            let adaptive_threshold = (self.noise_floor * 3.0).max(SILENCE_THRESHOLD * 0.5);

            if rms >= adaptive_threshold {
                // Found speech - update last speech position
                self.last_speech_pos = start_pos + (i + 1) * window_size;
            }
        }
    }

    pub fn take(&mut self) -> Vec<f32> {
        self.last_speech_pos = 0;
        self.overlap_buffer.clear();
        std::mem::take(&mut self.samples)
    }

    /// Check if there's been enough silence to warrant chunking
    /// Returns true if silence >= MIN_SILENCE_DURATION_MS since last speech
    pub fn has_silence_boundary(&self) -> bool {
        if self.samples.is_empty() || self.last_speech_pos == 0 {
            return false;
        }

        let silence_samples = self.samples.len().saturating_sub(self.last_speech_pos);
        let min_silence_samples = (self.sample_rate * MIN_SILENCE_DURATION_MS / 1000) as usize;

        silence_samples >= min_silence_samples
    }

    /// Take the audio chunk up to the silence boundary (speech portion only)
    /// Returns None if no silence boundary detected or buffer too small
    /// Includes overlap from previous chunk to prevent word cutting
    pub fn take_chunk_at_silence(&mut self) -> Option<Vec<f32>> {
        if !self.has_silence_boundary() {
            return None;
        }

        // Only take if we have meaningful content (at least 0.5s of speech)
        let min_chunk_samples = (self.sample_rate / 2) as usize;
        if self.last_speech_pos < min_chunk_samples {
            return None;
        }

        // Split at the middle of the silence for cleaner boundaries
        let silence_start = self.last_speech_pos;
        let silence_samples = self.samples.len() - silence_start;
        let split_point = silence_start + silence_samples / 2;

        // Calculate overlap size
        let overlap_samples = (self.sample_rate * CHUNK_OVERLAP_MS / 1000) as usize;

        // Create chunk with overlap from previous chunk prepended
        let mut chunk = Vec::with_capacity(self.overlap_buffer.len() + split_point);
        chunk.extend_from_slice(&self.overlap_buffer);
        chunk.extend_from_slice(&self.samples[..split_point]);

        // Save overlap for next chunk (last overlap_samples of current chunk)
        self.overlap_buffer.clear();
        if split_point > overlap_samples {
            self.overlap_buffer.extend_from_slice(&self.samples[split_point - overlap_samples..split_point]);
        }

        // Remove processed samples from buffer
        self.samples.drain(..split_point);

        // Reset speech position relative to remaining samples
        self.last_speech_pos = 0;

        log::info!(
            "Extracted chunk of {:.1}s (with {:.0}ms overlap), remaining buffer: {:.1}s, noise_floor: {:.4}",
            chunk.len() as f32 / self.sample_rate as f32,
            CHUNK_OVERLAP_MS,
            self.samples.len() as f32 / self.sample_rate as f32,
            self.noise_floor
        );

        Some(chunk)
    }

    /// Get current buffer length in samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the current adaptive noise floor
    pub fn get_noise_floor(&self) -> f32 {
        self.noise_floor
    }
}

/// Thread-safe audio buffer handle
pub type SharedBuffer = Arc<Mutex<AudioBuffer>>;

/// Creates a new shared buffer
pub fn create_shared_buffer() -> SharedBuffer {
    Arc::new(Mutex::new(AudioBuffer::new()))
}

/// Creates a new shared buffer with specified sample rate
pub fn create_shared_buffer_with_rate(sample_rate: u32) -> SharedBuffer {
    Arc::new(Mutex::new(AudioBuffer::with_sample_rate(sample_rate)))
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

/// Resample audio chunk to 16kHz for Whisper
/// Used for streaming transcription of individual chunks
pub fn resample_chunk(audio: &[f32], input_sample_rate: u32) -> Result<Vec<f32>, AudioError> {
    if input_sample_rate == WHISPER_SAMPLE_RATE {
        return Ok(audio.to_vec());
    }
    resample_audio(audio, input_sample_rate, WHISPER_SAMPLE_RATE)
}

/// Silence detection threshold (RMS value) - base threshold
const SILENCE_THRESHOLD: f32 = 0.01;
/// Minimum silence duration in milliseconds to be considered a split point
const MIN_SILENCE_DURATION_MS: u32 = 700; // Reduced from 1000ms for more responsive chunking
/// Minimum chunk duration in milliseconds
const MIN_CHUNK_DURATION_MS: u32 = 1000;

/// Calculate RMS (root mean square) of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Estimate noise floor from the beginning of the audio
fn estimate_noise_floor(audio: &[f32], sample_rate: u32) -> f32 {
    let window_size = (sample_rate / 50) as usize; // 20ms windows
    let num_windows = 25; // First 500ms

    let mut rms_values: Vec<f32> = Vec::new();
    for i in 0..num_windows {
        let start = i * window_size;
        if start + window_size <= audio.len() {
            rms_values.push(calculate_rms(&audio[start..start + window_size]));
        }
    }

    if rms_values.is_empty() {
        return SILENCE_THRESHOLD;
    }

    // Use the 10th percentile as noise floor estimate
    rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_idx = (rms_values.len() as f32 * 0.1) as usize;
    let noise_floor = rms_values.get(percentile_idx).copied().unwrap_or(SILENCE_THRESHOLD);

    // Return at least the minimum threshold
    noise_floor.max(SILENCE_THRESHOLD * 0.3)
}

/// Find silence boundaries in audio for chunking
/// Returns indices where the audio should be split (at the center of each silence gap)
/// Uses adaptive threshold based on estimated noise floor
pub fn find_silence_boundaries(audio: &[f32], sample_rate: u32) -> Vec<usize> {
    let min_silence_samples = (sample_rate * MIN_SILENCE_DURATION_MS / 1000) as usize;
    let min_chunk_samples = (sample_rate * MIN_CHUNK_DURATION_MS / 1000) as usize;
    let window_size = (sample_rate / 50) as usize; // 20ms windows for more stable RMS

    // Estimate noise floor for adaptive thresholding
    let noise_floor = estimate_noise_floor(audio, sample_rate);
    let adaptive_threshold = (noise_floor * 3.0).max(SILENCE_THRESHOLD * 0.5);

    log::info!(
        "Adaptive VAD: noise_floor={:.4}, threshold={:.4}",
        noise_floor,
        adaptive_threshold
    );

    let mut boundaries = Vec::new();
    let mut silence_start: Option<usize> = None;
    let mut last_boundary: usize = 0;

    let mut pos = 0;
    while pos + window_size <= audio.len() {
        let rms = calculate_rms(&audio[pos..pos + window_size]);

        if rms < adaptive_threshold {
            // In silence
            if silence_start.is_none() {
                silence_start = Some(pos);
            }
        } else {
            // Not in silence - check if we just exited a long enough silence
            if let Some(start) = silence_start {
                let silence_duration = pos - start;

                // Split at every silence >= threshold, but ensure minimum chunk size
                if silence_duration >= min_silence_samples {
                    let split_point = start + silence_duration / 2; // Split at middle of silence

                    // Only add boundary if it creates a chunk of at least min_chunk_samples
                    if split_point - last_boundary >= min_chunk_samples {
                        boundaries.push(split_point);
                        last_boundary = split_point;
                    }
                }
            }
            silence_start = None;
        }

        pos += window_size;
    }

    // Check for trailing silence (audio ends during silence)
    if let Some(start) = silence_start {
        let silence_duration = audio.len() - start;
        if silence_duration >= min_silence_samples {
            let split_point = start + silence_duration / 2;
            if split_point - last_boundary >= min_chunk_samples {
                boundaries.push(split_point);
            }
        }
    }

    log::info!(
        "Found {} silence boundaries in {} samples ({:.1}s audio)",
        boundaries.len(),
        audio.len(),
        audio.len() as f32 / sample_rate as f32
    );

    boundaries
}

/// Split audio at the given silence boundaries with overlap to prevent word cutting
/// If no boundaries, returns the entire audio as a single chunk
pub fn split_at_silences(audio: &[f32], boundaries: &[usize]) -> Vec<Vec<f32>> {
    split_at_silences_with_overlap(audio, boundaries, WHISPER_SAMPLE_RATE)
}

/// Split audio at the given silence boundaries with configurable overlap
pub fn split_at_silences_with_overlap(audio: &[f32], boundaries: &[usize], sample_rate: u32) -> Vec<Vec<f32>> {
    if boundaries.is_empty() {
        // No silence boundaries found - return entire audio as single chunk
        return vec![audio.to_vec()];
    }

    let overlap_samples = (sample_rate * CHUNK_OVERLAP_MS / 1000) as usize;
    let mut chunks = Vec::new();
    let mut start = 0;

    for &boundary in boundaries {
        if boundary > start && boundary < audio.len() {
            // Include overlap from previous chunk start (except for first chunk)
            let chunk_start = if start > 0 && start >= overlap_samples {
                start - overlap_samples
            } else {
                start
            };

            chunks.push(audio[chunk_start..boundary].to_vec());
            start = boundary;
        }
    }

    // Add the final chunk with overlap
    if start < audio.len() {
        let chunk_start = if start > 0 && start >= overlap_samples {
            start - overlap_samples
        } else {
            start
        };
        chunks.push(audio[chunk_start..].to_vec());
    }

    log::info!(
        "Split audio into {} chunks with {}ms overlap: {:?}",
        chunks.len(),
        CHUNK_OVERLAP_MS,
        chunks.iter().map(|c| format!("{:.1}s", c.len() as f32 / sample_rate as f32)).collect::<Vec<_>>()
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
    fn test_estimate_noise_floor() {
        let sample_rate = 16000;

        // Create audio with low background noise followed by speech
        let mut audio = Vec::new();

        // 500ms of low noise (noise floor estimation window)
        for i in 0..((0.5 * sample_rate as f32) as usize) {
            audio.push((i as f32 * 0.1).sin() * 0.002); // Very quiet
        }

        // Speech
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let noise_floor = estimate_noise_floor(&audio, sample_rate);

        // Noise floor should be detected as low
        assert!(noise_floor < 0.01, "Noise floor should be low, got {}", noise_floor);
    }

    #[test]
    fn test_find_silence_in_audio() {
        let sample_rate = 16000;
        // Create audio: 2s speech, 1s silence, 2s speech, 1s silence, 2s speech
        // (adjusted for new MIN_SILENCE_DURATION_MS = 700ms)
        let mut audio = Vec::new();

        // 2 seconds of "speech" (non-silent signal)
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 1 second of silence (should trigger split - >= 700ms)
        for _ in 0..(sample_rate as usize) {
            audio.push(0.0);
        }

        // 2 seconds of "speech"
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 1 second of silence
        for _ in 0..(sample_rate as usize) {
            audio.push(0.0);
        }

        // 2 seconds of "speech"
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find 2 boundaries (at each 1s silence)
        assert_eq!(boundaries.len(), 2, "Expected 2 boundaries, got {:?}", boundaries);
    }

    #[test]
    fn test_split_at_silences_with_overlap() {
        let sample_rate = 16000;
        // Create longer audio to test overlap properly
        let audio: Vec<f32> = (0..(sample_rate * 6)).map(|i| (i as f32 * 0.001).sin() * 0.1).collect();
        let boundaries = vec![sample_rate as usize * 2, sample_rate as usize * 4]; // Split at 2s and 4s

        let chunks = split_at_silences_with_overlap(&audio, &boundaries, sample_rate);

        assert_eq!(chunks.len(), 3);

        // First chunk: no overlap at start
        assert_eq!(chunks[0].len(), sample_rate as usize * 2);

        // Second chunk: includes overlap from previous
        let overlap_samples = (sample_rate * CHUNK_OVERLAP_MS / 1000) as usize;
        assert_eq!(chunks[1].len(), sample_rate as usize * 2 + overlap_samples);

        // Third chunk: includes overlap from previous
        assert_eq!(chunks[2].len(), sample_rate as usize * 2 + overlap_samples);
    }

    #[test]
    fn test_no_silence_returns_single_chunk() {
        let sample_rate = 16000;
        // Create 10 seconds of continuous speech with no silence
        let audio: Vec<f32> = (0..(10 * sample_rate))
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();

        let boundaries = find_silence_boundaries(&audio, sample_rate);
        let chunks = split_at_silences(&audio, &boundaries);

        // No silence found - should return entire audio as single chunk (no hard chunking)
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), audio.len());
    }

    #[test]
    fn test_audio_with_silence_is_chunked() {
        let sample_rate = 16000;
        // Create audio with a silence in the middle
        let mut audio = Vec::new();

        // 2 seconds of speech (>= MIN_CHUNK_DURATION_MS)
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 1 second of silence (>= 700ms threshold)
        for _ in 0..(sample_rate as usize) {
            audio.push(0.0);
        }

        // 2 seconds of speech
        for i in 0..(2 * sample_rate) {
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
        // Create audio with a short silence (< 700ms)
        let mut audio = Vec::new();

        // 2 seconds of speech
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 0.5 seconds of silence (< 700ms threshold)
        for _ in 0..((0.5 * sample_rate as f32) as usize) {
            audio.push(0.0);
        }

        // 2 seconds of speech
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should find no boundaries - silence too short
        assert!(boundaries.is_empty(), "Short silence should not trigger split");
    }

    #[test]
    fn test_adaptive_threshold_with_noisy_background() {
        let sample_rate = 16000;
        // Create audio with noisy background
        let mut audio = Vec::new();

        // 500ms of background noise (for noise floor estimation)
        for i in 0..((0.5 * sample_rate as f32) as usize) {
            audio.push((i as f32 * 0.1).sin() * 0.005); // Low noise
        }

        // 2 seconds of speech
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        // 1 second of "silence" (but with background noise)
        for i in 0..(sample_rate as usize) {
            audio.push((i as f32 * 0.1).sin() * 0.005); // Same low noise
        }

        // 2 seconds of speech
        for i in 0..(2 * sample_rate) {
            audio.push((i as f32 * 0.01).sin() * 0.3);
        }

        let boundaries = find_silence_boundaries(&audio, sample_rate);

        // Should detect the silence even with background noise (adaptive threshold)
        assert_eq!(boundaries.len(), 1, "Expected 1 boundary with adaptive threshold, got {:?}", boundaries);
    }

    #[test]
    fn test_audio_buffer_overlap() {
        let sample_rate = 16000;
        let mut buffer = AudioBuffer::with_sample_rate(sample_rate);

        // Push 3 seconds of speech
        let speech: Vec<f32> = (0..(3 * sample_rate))
            .map(|i| (i as f32 * 0.01).sin() * 0.3)
            .collect();
        buffer.push_samples(&speech);

        // Push 1.5 seconds of silence (trigger chunking)
        let silence: Vec<f32> = vec![0.0; (1.5 * sample_rate as f32) as usize];
        buffer.push_samples(&silence);

        // Should detect silence boundary
        assert!(buffer.has_silence_boundary());

        // Take chunk
        let chunk = buffer.take_chunk_at_silence();
        assert!(chunk.is_some());

        // Verify overlap buffer is populated
        let overlap_samples = (sample_rate * CHUNK_OVERLAP_MS / 1000) as usize;
        assert_eq!(buffer.overlap_buffer.len(), overlap_samples);
    }
}
