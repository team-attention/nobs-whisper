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
    ) -> Result<String, WhisperError> {
        let context = self.context.as_ref().ok_or(WhisperError::NoModel)?;

        log::info!(
            "Transcribing {} samples (~{:.1}s), language: {:?}",
            audio.len(),
            audio.len() as f32 / 16000.0,
            language
        );

        // Create state for this transcription
        let mut state = context
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

        // Set custom vocabulary as initial prompt to help recognize specific terms
        if let Some(vocab) = vocabulary {
            if !vocab.is_empty() {
                params.set_initial_prompt(vocab);
                log::debug!("Using custom vocabulary prompt");
            }
        }

        // Disable various features for faster inference
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_translate(false);
        params.set_no_context(true);
        params.set_single_segment(false);

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

        for (i, chunk) in chunks.iter().enumerate() {
            log::info!(
                "Transcribing chunk {}/{} ({:.1}s)",
                i + 1,
                chunks.len(),
                chunk.len() as f32 / 16000.0
            );

            match self.transcribe(chunk, language, vocabulary) {
                Ok(text) => {
                    if !text.is_empty() {
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
        let result = engine.transcribe(&[0.0; 16000], None, None);
        assert!(matches!(result, Err(WhisperError::NoModel)));
    }
}
