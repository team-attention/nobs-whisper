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
        ModelInfo {
            id: "tiny".to_string(),
            name: "Tiny".to_string(),
            size: 75_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fastest, lowest accuracy (~75MB)".to_string(),
        },
        ModelInfo {
            id: "base".to_string(),
            name: "Base".to_string(),
            size: 150_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fast, basic accuracy (~150MB)".to_string(),
        },
        ModelInfo {
            id: "small".to_string(),
            name: "Small".to_string(),
            size: 500_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Balanced performance (Recommended, ~500MB)".to_string(),
        },
        ModelInfo {
            id: "medium".to_string(),
            name: "Medium".to_string(),
            size: 1_500_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "High accuracy (~1.5GB)".to_string(),
        },
        ModelInfo {
            id: "large-v3".to_string(),
            name: "Large V3".to_string(),
            size: 3_000_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Best accuracy (~3GB)".to_string(),
        },
        ModelInfo {
            id: "large-v3-turbo".to_string(),
            name: "Large V3 Turbo".to_string(),
            size: 1_600_000_000,
            status: ModelStatus::NotDownloaded,
            download_progress: None,
            local_path: None,
            description: "Fast Large model (~1.6GB)".to_string(),
        },
    ]
}

fn model_url(model_id: &str) -> String {
    format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin",
        model_id
    )
}

fn model_path(model_id: &str) -> Result<PathBuf, ModelError> {
    let models_dir = AppConfig::models_dir()
        .map_err(|e| ModelError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())))?;
    Ok(models_dir.join(format!("ggml-{}.bin", model_id)))
}

#[tauri::command]
pub fn list_available_models() -> Vec<ModelInfo> {
    let mut models = available_models();
    let models_dir = match AppConfig::models_dir() {
        Ok(dir) => dir,
        Err(_) => return models,
    };

    for model in &mut models {
        let path = models_dir.join(format!("ggml-{}.bin", model.id));
        if path.exists() {
            model.status = ModelStatus::Downloaded;
            model.local_path = Some(path.to_string_lossy().to_string());
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

    let url = model_url(&model_id);
    let path = model_path(&model_id).map_err(|e| e.to_string())?;

    // Mark as downloading
    {
        let mut progress = DOWNLOAD_PROGRESS.lock().await;
        progress.insert(model_id.clone(), 0.0);
    }

    // Download the model
    let result = download_file(&url, &path, &model_id).await;

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

    let path = model_path(&model_id).map_err(|e| e.to_string())?;

    if path.exists() {
        fs::remove_file(&path).map_err(|e| e.to_string())?;
        log::info!("Model {} deleted", model_id);
    }

    Ok(())
}
