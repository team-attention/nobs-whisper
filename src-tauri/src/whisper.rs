use std::path::Path;
use thiserror::Error;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum WhisperError {
    #[error("Failed to load model: {0}")]
    LoadError(String),
    #[error("Transcription failed: {0}")]
    TranscriptionError(String),
    #[error("No model loaded")]
    NoModel,
}

pub struct WhisperEngine {
    context: Option<WhisperContext>,
    model_path: Option<String>,
}

impl WhisperEngine {
    pub fn new() -> Self {
        Self {
            context: None,
            model_path: None,
        }
    }

    /// Create a WhisperEngine and load a model from file
    pub fn from_file(model_path: &Path) -> Result<Self, WhisperError> {
        let mut engine = Self::new();
        engine.load_model(model_path)?;
        Ok(engine)
    }

    pub fn load_model(&mut self, model_path: &Path) -> Result<(), WhisperError> {
        log::info!("Loading Whisper model from: {:?}", model_path);

        let mut params = WhisperContextParameters::default();
        params.use_gpu(true);
        let context = WhisperContext::new_with_params(
            model_path.to_str().unwrap_or_default(),
            params,
        )
        .map_err(|e| WhisperError::LoadError(e.to_string()))?;

        self.context = Some(context);
        self.model_path = Some(model_path.to_string_lossy().to_string());

        log::info!("Model loaded successfully");
        Ok(())
    }

    #[allow(dead_code)]
    pub fn unload_model(&mut self) {
        self.context = None;
        self.model_path = None;
        log::info!("Model unloaded");
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
        vocabulary: Option<&str>,
        context: Option<&str>,
    ) -> Result<String, WhisperError> {
        let whisper_ctx = self.context.as_ref().ok_or(WhisperError::NoModel)?;

        log::info!(
            "Transcribing {} samples (~{:.1}s), language: {:?}",
            audio.len(),
            audio.len() as f32 / 16000.0,
            language
        );

        // Create state for this transcription
        let mut state = whisper_ctx
            .create_state()
            .map_err(|e| WhisperError::TranscriptionError(e.to_string()))?;

        // Configure transcription parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Set language
        if let Some(lang) = language {
            params.set_language(Some(lang));
        } else {
            params.set_language(None); // Auto-detect
        }

        // Build initial prompt from custom vocabulary and previous transcription context
        let initial_prompt = match (vocabulary, context) {
            (Some(vocab), Some(ctx)) if !vocab.is_empty() => {
                Some(format!("{} {}", vocab, ctx))
            }
            (Some(vocab), None) if !vocab.is_empty() => Some(vocab.to_string()),
            (_, Some(ctx)) => Some(ctx.to_string()),
            _ => None,
        };
        if let Some(ref prompt) = initial_prompt {
            params.set_initial_prompt(prompt);
            log::debug!("Using initial prompt ({} chars)", prompt.len());
        }

        // Disable various features for faster inference
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_no_context(false);
        params.set_single_segment(false);

        // Suppress hallucinations (e.g. "Thank you for watching!" from YouTube training data)
        params.set_suppress_blank(true);
        params.set_no_speech_thold(0.6);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);

        // Run transcription
        state
            .full(params, audio)
            .map_err(|e| WhisperError::TranscriptionError(e.to_string()))?;

        // Collect results
        let num_segments = state.full_n_segments();

        let mut result = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(text) = segment.to_str_lossy() {
                    result.push_str(&text);
                }
            }
        }

        let result = result.trim().to_string();
        let result = filter_hallucinations(&result);
        log::info!("Transcription complete: {} characters", result.len());

        Ok(result)
    }

    /// Transcribe multiple audio chunks and join results
    /// Each chunk is transcribed sequentially (GPU can only process one at a time)
    pub fn transcribe_chunked(
        &self,
        chunks: Vec<Vec<f32>>,
        language: Option<&str>,
        vocabulary: Option<&str>,
    ) -> Result<String, WhisperError> {
        log::info!(
            "Transcribing {} chunks, language: {:?}",
            chunks.len(),
            language
        );

        let mut results = Vec::new();
        let mut last_context: Option<String> = None;

        for (i, chunk) in chunks.iter().enumerate() {
            log::info!(
                "Transcribing chunk {}/{} ({:.1}s)",
                i + 1,
                chunks.len(),
                chunk.len() as f32 / 16000.0
            );

            match self.transcribe(chunk, language, vocabulary, last_context.as_deref()) {
                Ok(text) => {
                    if !text.is_empty() {
                        last_context = Some(text.clone());
                        results.push(text);
                    }
                }
                Err(e) => {
                    log::error!("Failed to transcribe chunk {}: {}", i + 1, e);
                    return Err(e);
                }
            }
        }

        let combined = results.join(" ");
        log::info!(
            "Chunked transcription complete: {} characters from {} chunks",
            combined.len(),
            chunks.len()
        );

        Ok(combined)
    }
}

/// Known Whisper hallucination phrases (from YouTube training data).
/// These appear during silence or low-energy audio segments.
const HALLUCINATION_PHRASES: &[&str] = &[
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "see you next time",
    "please like and subscribe",
    "don't forget to subscribe",
    "hit the bell",
    "leave a comment",
    "check out my other videos",
    "thanks for tuning in",
    // Korean equivalents
    "시청해 주셔서 감사합니다",
    "구독과 좋아요",
    "구독 부탁드립니다",
    // Japanese equivalents
    "ご視聴ありがとうございました",
    // Chinese equivalents
    "感谢收看",
    "谢谢观看",
    // Common short hallucinations
    "you",
    "MBC 뉴스 이덕영입니다",
];

/// Filter out known Whisper hallucination phrases from transcription output.
fn filter_hallucinations(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Discard output that is only punctuation/symbols (e.g. "...", "♪", "...")
    if trimmed.chars().all(|c| c.is_ascii_punctuation() || matches!(c, '…' | '♪' | '\u{266B}' | '\u{266C}')) {
        log::info!("Filtered punctuation-only hallucination: {:?}", trimmed);
        return String::new();
    }

    let lower = trimmed.to_lowercase();

    // If the entire output is a known hallucination phrase, discard it
    for phrase in HALLUCINATION_PHRASES {
        let pattern = phrase.to_lowercase();
        // Exact match (ignoring trailing punctuation like .!?)
        let stripped = lower
            .trim_end_matches(|c: char| c.is_ascii_punctuation() || matches!(c, '…' | '♪'));
        if stripped == pattern {
            log::info!("Filtered hallucination: {:?}", trimmed);
            return String::new();
        }
    }

    trimmed.to_string()
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_engine_new() {
        let engine = WhisperEngine::new();
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_transcribe_without_model() {
        let engine = WhisperEngine::new();
        let result = engine.transcribe(&[0.0; 16000], None, None, None);
        assert!(matches!(result, Err(WhisperError::NoModel)));
    }

    #[test]
    fn test_filter_hallucinations() {
        // Known hallucinations should be filtered
        assert_eq!(filter_hallucinations("Thank you for watching!"), "");
        assert_eq!(filter_hallucinations("thanks for watching."), "");
        assert_eq!(filter_hallucinations("Thank you for watching"), "");
        assert_eq!(filter_hallucinations("Subscribe to my channel"), "");
        assert_eq!(filter_hallucinations("you"), "");
        assert_eq!(filter_hallucinations("..."), "");
        assert_eq!(filter_hallucinations("시청해 주셔서 감사합니다"), "");

        // Real content should pass through
        assert_eq!(
            filter_hallucinations("Hello, this is a real sentence."),
            "Hello, this is a real sentence."
        );
        assert_eq!(
            filter_hallucinations("Thank you for watching the demo, now let me explain"),
            "Thank you for watching the demo, now let me explain"
        );
    }
}
