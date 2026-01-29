use crate::config::AppConfig;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::LazyLock;
use tokio::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Model not found: {0}")]
    NotFound(String),
    #[error("Download in progress")]
    DownloadInProgress,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub size: u64,
    pub status: ModelStatus,
    pub download_progress: Option<f64>,
    pub local_path: Option<String>,
    pub description: String,
    pub category: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    NotDownloaded,
    Downloading,
    Downloaded,
}

static DOWNLOAD_PROGRESS: LazyLock<Mutex<HashMap<String, f64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn available_models() -> Vec<ModelInfo> {
    vec![
        // Official Whisper models from ggerganov/whisper.cpp
        ModelInfo {
            id: "tiny".to_string(),
            name: "Tiny".to_string(),
            size: 75_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fastest, lowest accuracy (~75MB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".to_string(),
        },
        ModelInfo {
            id: "base".to_string(),
            name: "Base".to_string(),
            size: 150_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fast, basic accuracy (~150MB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".to_string(),
        },
        ModelInfo {
            id: "small".to_string(),
            name: "Small".to_string(),
            size: 500_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Balanced performance (Recommended, ~500MB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".to_string(),
        },
        ModelInfo {
            id: "medium".to_string(),
            name: "Medium".to_string(),
            size: 1_500_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "High accuracy (~1.5GB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".to_string(),
        },
        ModelInfo {
            id: "large-v3".to_string(),
            name: "Large V3".to_string(),
            size: 3_000_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Best accuracy (~3GB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin".to_string(),
        },
        ModelInfo {
            id: "large-v3-turbo".to_string(),
            name: "Large V3 Turbo".to_string(),
            size: 1_600_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fast Large model (~1.6GB)".to_string(),
            category: "Official".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin".to_string(),
        },
        // Distil-Whisper models - faster distilled versions (English-only)
        ModelInfo {
            id: "distil-small.en".to_string(),
            name: "Distil Small (EN)".to_string(),
            size: 340_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "6x faster than small, English-only (~340MB)".to_string(),
            category: "Distil-Whisper".to_string(),
            url: "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin".to_string(),
        },
        ModelInfo {
            id: "distil-medium.en".to_string(),
            name: "Distil Medium (EN)".to_string(),
            size: 770_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "6x faster than medium, English-only (~770MB)".to_string(),
            category: "Distil-Whisper".to_string(),
            url: "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-distil-medium.en.bin".to_string(),
        },
        ModelInfo {
            id: "distil-large-v3".to_string(),
            name: "Distil Large V3".to_string(),
            size: 1_500_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "5x faster than large-v3, multilingual (~1.5GB)".to_string(),
            category: "Distil-Whisper".to_string(),
            url: "https://huggingface.co/distil-whisper/distil-large-v3/resolve/main/ggml-distil-large-v3.bin".to_string(),
        },
        // Quantized models - smaller file sizes with minimal quality loss
        ModelInfo {
            id: "small-q5_1".to_string(),
            name: "Small Q5_1".to_string(),
            size: 190_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Quantized small, 60% smaller (~190MB)".to_string(),
            category: "Quantized".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin".to_string(),
        },
        ModelInfo {
            id: "medium-q5_0".to_string(),
            name: "Medium Q5_0".to_string(),
            size: 540_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Quantized medium, 65% smaller (~540MB)".to_string(),
            category: "Quantized".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin".to_string(),
        },
        ModelInfo {
            id: "large-v3-q5_0".to_string(),
            name: "Large V3 Q5_0".to_string(),
            size: 1_100_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Quantized large-v3, 65% smaller (~1.1GB)".to_string(),
            category: "Quantized".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-q5_0.bin".to_string(),
        },
    ]
}

fn model_path(model: &ModelInfo) -> Result<PathBuf, ModelError> {
    let models_dir = AppConfig::models_dir()
        .map_err(|e| ModelError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    // Extract filename from URL or use model id
    let default_filename = format!("ggml-{}.bin", model.id);
    let filename = model.url.rsplit('/').next()
        .unwrap_or(&default_filename);
    Ok(models_dir.join(filename))
}

fn model_path_by_id(model_id: &str) -> Result<PathBuf, ModelError> {
    let models = available_models();
    let model = models.iter().find(|m| m.id == model_id)
        .ok_or_else(|| ModelError::NotFound(model_id.to_string()))?;
    model_path(model)
}

#[tauri::command]
pub fn list_available_models() -> Vec<ModelInfo> {
    let mut models = available_models();

    for model in &mut models {
        if let Ok(path) = model_path(model) {
            if path.exists() {
                model.status = ModelStatus::Downloaded;
                model.local_path = Some(path.to_string_lossy().to_string());
            }
        }
    }

    models
}

#[tauri::command]
pub fn list_downloaded_models() -> Vec<ModelInfo> {
    list_available_models()
        .into_iter()
        .filter(|m| m.status == ModelStatus::Downloaded)
        .collect()
}

#[tauri::command]
pub async fn download_model(model_id: String) -> Result<ModelInfo, String> {
    log::info!("Starting download for model: {}", model_id);

    // Check if already downloading
    {
        let progress = DOWNLOAD_PROGRESS.lock().await;
        if progress.contains_key(&model_id) {
            return Err("Download already in progress".to_string());
        }
    }

    // Find model info
    let models = available_models();
    let model = models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Model not found: {}", model_id))?
        .clone();

    let url = &model.url;
    let path = model_path(&model).map_err(|e| e.to_string())?;

    // Mark as downloading
    {
        let mut progress = DOWNLOAD_PROGRESS.lock().await;
        progress.insert(model_id.clone(), 0.0);
    }

    // Download the model
    let result = download_file(url, &path, &model_id).await;

    // Clear progress
    {
        let mut progress = DOWNLOAD_PROGRESS.lock().await;
        progress.remove(&model_id);
    }

    match result {
        Ok(()) => {
            log::info!("Model {} downloaded successfully", model_id);
            Ok(ModelInfo {
                id: model.id,
                name: model.name,
                size: model.size,
                status: ModelStatus::Downloaded,
                download_progress: None,
                local_path: Some(path.to_string_lossy().to_string()),
                description: model.description,
                category: model.category,
                url: model.url,
            })
        }
        Err(e) => {
            log::error!("Failed to download model {}: {}", model_id, e);
            // Clean up partial download
            let _ = fs::remove_file(&path);
            Err(e.to_string())
        }
    }
}

async fn download_file(url: &str, path: &PathBuf, model_id: &str) -> Result<(), ModelError> {
    let client = reqwest::Client::new();
    let response = client.get(url).send().await?;

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;

    let mut file = fs::File::create(path)?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;

        // Update progress
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64) * 100.0;
            let mut progress_map = DOWNLOAD_PROGRESS.lock().await;
            progress_map.insert(model_id.to_string(), progress);
        }
    }

    file.flush()?;
    Ok(())
}

#[tauri::command]
pub async fn get_download_progress(model_id: String) -> Option<f64> {
    let progress = DOWNLOAD_PROGRESS.lock().await;
    progress.get(&model_id).copied()
}

#[tauri::command]
pub fn delete_model(model_id: String) -> Result<(), String> {
    log::info!("Deleting model: {}", model_id);

    let path = model_path_by_id(&model_id).map_err(|e| e.to_string())?;

    if path.exists() {
        fs::remove_file(&path).map_err(|e| e.to_string())?;
        log::info!("Model {} deleted", model_id);
    }

    Ok(())
}
