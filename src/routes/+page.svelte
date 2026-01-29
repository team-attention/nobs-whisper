<script lang="ts">
  import { invoke, isTauri as checkIsTauri } from "@tauri-apps/api/core";
  import { onMount } from "svelte";

  let isTauri = $state(false);

  interface ModelInfo {
    id: string;
    name: string;
    size: number;
    status: "not_downloaded" | "downloading" | "downloaded";
    downloadProgress?: number;
    localPath?: string;
    description: string;
    category: string;
    url: string;
  }

  interface AppConfig {
    selectedModel: string | null;
    shortcut: string;
    language: string;
    autoLaunch: boolean;
    maxRecordingDuration: number;
  }

  let models = $state<ModelInfo[]>([]);
  let config = $state<AppConfig>({
    selectedModel: null,
    shortcut: "Ctrl+Alt+Space",
    language: "auto",
    autoLaunch: false,
    maxRecordingDuration: 60,
  });
  let downloadingModel = $state<string | null>(null);
  let downloadProgress = $state<number>(0);
  let deletingModel = $state<string | null>(null);
  let hasAccessibility = $state<boolean>(true);
  let error = $state<string | null>(null);
  let modelExpanded = $state(false);

  // Shortcut capture state
  let isCapturingShortcut = $state(false);
  let capturedShortcut = $state("");
  let captureElement: HTMLDivElement | null = $state(null);

  const languages = [
    { value: "auto", label: "Auto-detect" },
    { value: "ko", label: "Korean" },
    { value: "en", label: "English" },
    { value: "ja", label: "Japanese" },
    { value: "zh", label: "Chinese" },
    { value: "es", label: "Spanish" },
    { value: "fr", label: "French" },
    { value: "de", label: "German" },
  ];

  let selectedModelInfo = $derived(models.find(m => m.id === config.selectedModel));

  // Group models by category
  let modelsByCategory = $derived(() => {
    const categories = new Map<string, ModelInfo[]>();
    const order = ["Official", "Distil-Whisper", "Quantized"];

    for (const model of models) {
      const cat = model.category || "Other";
      if (!categories.has(cat)) {
        categories.set(cat, []);
      }
      categories.get(cat)!.push(model);
    }

    // Sort by predefined order
    const sorted = new Map<string, ModelInfo[]>();
    for (const cat of order) {
      if (categories.has(cat)) {
        sorted.set(cat, categories.get(cat)!);
        categories.delete(cat);
      }
    }
    // Add any remaining categories
    for (const [cat, models] of categories) {
      sorted.set(cat, models);
    }

    return sorted;
  });

  let expandedCategories = $state<Set<string>>(new Set(["Official"]));

  onMount(() => {
    isTauri = checkIsTauri();

    if (!isTauri) {
      error = "Not running in Tauri context.";
      return;
    }

    loadData().catch((e) => {
      console.error("Failed to load data:", e);
      error = `Failed to load data: ${e}`;
    });

    const interval = setInterval(async () => {
      if (downloadingModel) {
        const progress = await invoke<number | null>("get_download_progress", {
          modelId: downloadingModel,
        });
        if (progress !== null) {
          downloadProgress = progress;
        } else {
          downloadingModel = null;
          downloadProgress = 0;
          await loadModels();
        }
      }
    }, 500);

    return () => {
      clearInterval(interval);
    };
  });

  async function loadData() {
    await Promise.all([loadModels(), loadConfig(), checkAccessibility()]);
  }

  async function loadModels() {
    if (!isTauri) return;
    try {
      models = await invoke<ModelInfo[]>("list_available_models");
    } catch (e) {
      console.error("Failed to load models:", e);
      error = `Failed to load models: ${e}`;
    }
  }

  async function loadConfig() {
    if (!isTauri) return;
    try {
      config = await invoke<AppConfig>("get_config");
    } catch (e) {
      console.error("Failed to load config:", e);
    }
  }

  async function checkAccessibility() {
    if (!isTauri) return;
    try {
      hasAccessibility = await invoke<boolean>("check_accessibility_permission");
    } catch (e) {
      console.error("Failed to check accessibility:", e);
    }
  }

  async function saveConfig() {
    if (!isTauri) return;
    try {
      await invoke("set_config", { config });
      error = null;
    } catch (e) {
      console.error("Failed to save config:", e);
      error = `Failed to save config: ${e}`;
    }
  }

  async function downloadModel(modelId: string) {
    if (!isTauri || downloadingModel) return;

    downloadingModel = modelId;
    downloadProgress = 0;

    try {
      await invoke<ModelInfo>("download_model", { modelId });
      await loadModels();
    } catch (e) {
      console.error("Failed to download model:", e);
      error = `Failed to download model: ${e}`;
    } finally {
      downloadingModel = null;
      downloadProgress = 0;
    }
  }

  async function deleteModel(modelId: string) {
    if (!isTauri || deletingModel) return;

    deletingModel = modelId;

    try {
      await invoke("delete_model", { modelId });

      // Brief delay for animation
      await new Promise(r => setTimeout(r, 300));

      await loadModels();

      if (config.selectedModel === modelId) {
        config.selectedModel = null;
        await saveConfig();
      }
    } catch (e) {
      console.error("Failed to delete model:", e);
      error = `Failed to delete model: ${e}`;
    } finally {
      deletingModel = null;
    }
  }

  async function selectModel(modelId: string) {
    config.selectedModel = modelId;
    await saveConfig();
    modelExpanded = false;
  }

  function formatSize(bytes: number): string {
    if (bytes >= 1_000_000_000) {
      return `${(bytes / 1_000_000_000).toFixed(1)}GB`;
    }
    return `${(bytes / 1_000_000).toFixed(0)}MB`;
  }

  // Shortcut capture functions
  async function startCapture() {
    if (isTauri) {
      try {
        console.log("Pausing shortcuts before capture...");
        await invoke("unregister_current_shortcut");
        await invoke("pause_native_listener");
        console.log("Shortcuts paused successfully");
      } catch (e) {
        console.error("Failed to pause shortcuts:", e);
      }
    }
    isCapturingShortcut = true;
    capturedShortcut = "";
    setTimeout(() => captureElement?.focus(), 0);
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (!isCapturingShortcut) return;

    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();

    if (e.key === "Alt" || e.key === "Option") {
      capturedShortcut = e.location === 2 ? "RightOption" : "LeftOption";
      return;
    }
    if (e.key === "Meta") {
      capturedShortcut = e.location === 2 ? "RightCmd" : "LeftCmd";
      return;
    }
    if (e.key === "Control") {
      capturedShortcut = e.location === 2 ? "RightCtrl" : "LeftCtrl";
      return;
    }
    if (e.key === "Shift") {
      capturedShortcut = e.location === 2 ? "RightShift" : "LeftShift";
      return;
    }

    const keys: string[] = [];
    if (e.metaKey) keys.push("Cmd");
    if (e.ctrlKey) keys.push("Ctrl");
    if (e.altKey) keys.push("Alt");
    if (e.shiftKey) keys.push("Shift");

    const key = e.key;
    if (!["Meta", "Control", "Alt", "Shift"].includes(key)) {
      const normalizedKey = key === " " ? "Space" : key.length === 1 ? key.toUpperCase() : key;
      keys.push(normalizedKey);
    }

    if (keys.length > 0) {
      capturedShortcut = keys.join("+");
    }
  }

  async function confirmCapture() {
    if (!capturedShortcut) {
      error = "Please press a key combination";
      return;
    }

    const isNative = isNativeShortcut(capturedShortcut);
    if (!isNative && !capturedShortcut.includes("+")) {
      error = "Please press a modifier key (Cmd/Ctrl/Alt/Shift) with another key";
      return;
    }

    config.shortcut = capturedShortcut;
    try {
      await saveConfig();
      isCapturingShortcut = false;
      capturedShortcut = "";
      await invoke("resume_native_listener");
    } catch (e) {
      error = `Invalid shortcut: ${e}`;
      await invoke("resume_native_listener");
    }
  }

  async function cancelCapture() {
    isCapturingShortcut = false;
    capturedShortcut = "";
    if (isTauri) {
      try {
        await invoke("register_current_shortcut");
        await invoke("resume_native_listener");
      } catch (e) {
        console.error("Failed to resume shortcuts:", e);
      }
    }
  }

  const nativeShortcuts = [
    "RightOption",
    "LeftOption",
    "RightCmd",
    "LeftCmd",
    "RightCtrl",
    "LeftCtrl",
    "RightShift",
    "LeftShift",
  ];

  function isNativeShortcut(shortcut: string): boolean {
    return nativeShortcuts.some(
      (s) => s.toLowerCase() === shortcut.toLowerCase()
    );
  }

</script>

<main>
  {#if !isTauri}
    <div class="error">Not running in Tauri context.</div>
  {:else}
    {#if !hasAccessibility}
      <div class="warning">
        Accessibility permission required for text input.
      </div>
    {/if}
    {#if error}
      <div class="error">{error}</div>
    {/if}

    <div class="group">
      <label>Model</label>
      <div class="model-selector">
        <button
          class="model-header"
          onclick={() => modelExpanded = !modelExpanded}
        >
          <span class="model-selected">
            {#if selectedModelInfo}
              {selectedModelInfo.name}
            {:else}
              Select a model
            {/if}
          </span>
          <span class="chevron" class:expanded={modelExpanded}>▼</span>
        </button>

        {#if modelExpanded}
          <div class="model-list">
            {#each [...modelsByCategory()] as [category, categoryModels]}
              <div class="category-section">
                <button
                  class="category-header"
                  onclick={() => {
                    const newSet = new Set(expandedCategories);
                    if (newSet.has(category)) {
                      newSet.delete(category);
                    } else {
                      newSet.add(category);
                    }
                    expandedCategories = newSet;
                  }}
                >
                  <span class="category-chevron" class:expanded={expandedCategories.has(category)}>▶</span>
                  <span class="category-name">{category}</span>
                  <span class="category-count">{categoryModels.length}</span>
                </button>
                {#if expandedCategories.has(category)}
                  <div class="category-models">
                    {#each categoryModels as model}
                      <div
                        class="model-item"
                        class:selected={model.id === config.selectedModel}
                        class:downloading={downloadingModel === model.id}
                        class:deleting={deletingModel === model.id}
                      >
                        {#if downloadingModel === model.id}
                          <div class="download-progress-bar" style="width: {downloadProgress}%"></div>
                        {/if}
                        <button
                          class="model-info"
                          onclick={() => model.status === "downloaded" && selectModel(model.id)}
                          disabled={model.status !== "downloaded" || deletingModel === model.id}
                        >
                          <span class="model-name">{model.name}</span>
                          <span class="model-size">{formatSize(model.size)}</span>
                        </button>
                        <div class="model-actions">
                          {#if deletingModel === model.id}
                            <span class="model-status">Deleting...</span>
                          {:else if model.status === "downloaded"}
                            {#if model.id === config.selectedModel}
                              <span class="model-badge">Selected</span>
                            {/if}
                            <button
                              class="btn-action delete"
                              onclick={() => deleteModel(model.id)}
                            >
                              Delete
                            </button>
                          {:else if downloadingModel === model.id}
                            <span class="model-progress">{downloadProgress.toFixed(0)}%</span>
                          {:else}
                            <button
                              class="btn-action download"
                              onclick={() => downloadModel(model.id)}
                              disabled={downloadingModel !== null}
                            >
                              Download
                            </button>
                          {/if}
                        </div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <div class="group">
      <label>Shortcut</label>
      {#if isCapturingShortcut}
        <div
          class="shortcut-capture"
          bind:this={captureElement}
          onkeydown={handleKeyDown}
          tabindex="0"
          role="button"
        >
          <span class="capture-display">
            {capturedShortcut || "Press keys..."}
          </span>
          <div class="capture-actions">
            <button class="btn-text" onclick={confirmCapture} disabled={!capturedShortcut}>
              Confirm
            </button>
            <button class="btn-text" onclick={cancelCapture}>
              Cancel
            </button>
          </div>
        </div>
      {:else}
        <div class="shortcut-display">
          <span class="shortcut-value">{config.shortcut}</span>
          <button class="btn-text" onclick={startCapture}>Change</button>
        </div>
        {#if isNativeShortcut(config.shortcut)}
          <div class="hint">Press to toggle recording</div>
        {/if}
      {/if}
    </div>

    <div class="group">
      <label for="language">Language</label>
      <select id="language" bind:value={config.language} onchange={saveConfig}>
        {#each languages as lang}
          <option value={lang.value}>{lang.label}</option>
        {/each}
      </select>
    </div>
  {/if}
</main>

<style>
  :global(html) {
    scrollbar-gutter: stable;
  }

  :global(body) {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    font-size: 13px;
    line-height: 1.4;
    color: #1d1d1f;
    background: #fafafa;
    margin: 0;
    padding: 0;
  }

  main {
    padding: 24px;
    max-width: 400px;
    margin: 0 auto;
  }

  .warning {
    background: #f5f5f5;
    border-left: 2px solid #999;
    padding: 8px 12px;
    margin-bottom: 20px;
    font-size: 12px;
    color: #666;
  }

  .error {
    background: #f5f5f5;
    border-left: 2px solid #666;
    padding: 8px 12px;
    margin-bottom: 20px;
    font-size: 12px;
    color: #333;
  }

  .group {
    margin-bottom: 20px;
  }

  label {
    display: block;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #666;
    margin-bottom: 6px;
  }

  select {
    width: 100%;
    padding: 10px 12px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    font-size: 13px;
    background: #fff;
    color: #1d1d1f;
    box-sizing: border-box;
    transition: border-color 0.15s;
  }

  select:focus {
    outline: none;
    border-color: #999;
  }

  /* Model Selector */
  .model-selector {
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
  }

  .model-header {
    width: 100%;
    padding: 10px 12px;
    border: none;
    background: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    font-size: 13px;
    color: #1d1d1f;
    text-align: left;
  }

  .model-header:hover {
    background: #f5f5f5;
  }

  .model-selected {
    font-weight: 500;
  }

  .chevron {
    font-size: 10px;
    color: #999;
    transition: transform 0.2s;
  }

  .chevron.expanded {
    transform: rotate(180deg);
  }

  .model-list {
    border-top: 1px solid #e0e0e0;
  }

  .category-section {
    border-bottom: 1px solid #e0e0e0;
  }

  .category-section:last-child {
    border-bottom: none;
  }

  .category-header {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: #f8f8f8;
    border: none;
    cursor: pointer;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #666;
    text-align: left;
  }

  .category-header:hover {
    background: #f0f0f0;
  }

  .category-chevron {
    font-size: 8px;
    transition: transform 0.2s;
  }

  .category-chevron.expanded {
    transform: rotate(90deg);
  }

  .category-name {
    flex: 1;
  }

  .category-count {
    font-weight: 400;
    color: #999;
  }

  .category-models {
    /* Container for models within a category */
  }

  .model-item {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid #f0f0f0;
    position: relative;
    overflow: hidden;
  }

  .model-item:last-child {
    border-bottom: none;
  }

  .model-item.selected {
    background: #f8f8f8;
  }

  .model-item.downloading {
    background: #fafafa;
  }

  .model-item.deleting {
    opacity: 0.5;
    background: linear-gradient(90deg, #fff 0%, #fee 50%, #fff 100%);
    background-size: 200% 100%;
    animation: deleting-pulse 0.6s ease-in-out infinite;
  }

  @keyframes deleting-pulse {
    0%, 100% { background-position: 0% 0%; }
    50% { background-position: 100% 0%; }
  }

  .model-status {
    font-size: 11px;
    color: #999;
    font-style: italic;
  }

  .download-progress-bar {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    background: linear-gradient(90deg, rgba(0, 122, 255, 0.1) 0%, rgba(0, 122, 255, 0.15) 100%);
    transition: width 0.3s ease;
    z-index: 0;
  }

  .model-info {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-align: left;
    position: relative;
    z-index: 1;
  }

  .model-info:disabled {
    cursor: default;
    opacity: 0.6;
  }

  .model-name {
    font-size: 13px;
    color: #1d1d1f;
  }

  .model-size {
    font-size: 11px;
    color: #999;
  }

  .model-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    position: relative;
    z-index: 1;
  }

  .model-badge {
    font-size: 10px;
    color: #666;
    background: #e8e8e8;
    padding: 2px 6px;
    border-radius: 3px;
  }

  .model-progress {
    font-size: 11px;
    color: #666;
    min-width: 36px;
    text-align: right;
  }

  .btn-action {
    background: none;
    border: 1px solid #e0e0e0;
    padding: 4px 10px;
    font-size: 11px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-action.download {
    color: #333;
  }

  .btn-action.download:hover:not(:disabled) {
    background: #f0f0f0;
    border-color: #ccc;
  }

  .btn-action.delete {
    color: #999;
  }

  .btn-action.delete:hover {
    color: #c00;
    border-color: #c00;
  }

  .btn-action:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .btn-text {
    background: none;
    border: none;
    padding: 4px 8px;
    font-size: 12px;
    color: #666;
    cursor: pointer;
    transition: color 0.15s;
  }

  .btn-text:hover:not(:disabled) {
    color: #1d1d1f;
  }

  .btn-text:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .shortcut-capture {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    border: 1px solid #999;
    border-radius: 6px;
    background: #fff;
    outline: none;
  }

  .shortcut-capture:focus {
    border-color: #666;
  }

  .capture-display {
    font-size: 14px;
    font-weight: 500;
    text-align: center;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 4px;
    min-height: 20px;
  }

  .capture-actions {
    display: flex;
    justify-content: center;
    gap: 12px;
  }

  .shortcut-display {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background: #fff;
  }

  .shortcut-value {
    font-size: 13px;
    font-weight: 500;
  }

  .hint {
    margin-top: 8px;
    font-size: 11px;
    color: #999;
    font-style: italic;
  }

  @media (prefers-color-scheme: dark) {
    :global(body) {
      color: #f5f5f7;
      background: #1c1c1e;
    }

    select {
      background: #2c2c2e;
      border-color: #3a3a3c;
      color: #f5f5f7;
    }

    select:focus {
      border-color: #666;
    }

    .model-selector {
      background: #2c2c2e;
      border-color: #3a3a3c;
    }

    .model-header {
      color: #f5f5f7;
    }

    .model-header:hover {
      background: #3a3a3c;
    }

    .model-list {
      border-color: #3a3a3c;
    }

    .category-section {
      border-color: #3a3a3c;
    }

    .category-header {
      background: #2a2a2c;
      color: #999;
    }

    .category-header:hover {
      background: #3a3a3c;
    }

    .model-item {
      border-color: #3a3a3c;
    }

    .model-item.selected {
      background: #3a3a3c;
    }

    .model-item.downloading {
      background: #2a2a2c;
    }

    .model-item.deleting {
      background: linear-gradient(90deg, #2c2c2e 0%, #3a2a2a 50%, #2c2c2e 100%);
      background-size: 200% 100%;
    }

    .download-progress-bar {
      background: linear-gradient(90deg, rgba(10, 132, 255, 0.2) 0%, rgba(10, 132, 255, 0.25) 100%);
    }

    .model-name {
      color: #f5f5f7;
    }

    .model-badge {
      background: #3a3a3c;
      color: #999;
    }

    .btn-action {
      border-color: #3a3a3c;
    }

    .btn-action.download {
      color: #ccc;
    }

    .btn-action.download:hover:not(:disabled) {
      background: #3a3a3c;
      border-color: #666;
    }

    .btn-action.delete {
      color: #666;
    }

    .btn-action.delete:hover {
      color: #ff6b6b;
      border-color: #ff6b6b;
    }

    .warning, .error {
      background: #2c2c2e;
    }

    .error {
      border-color: #666;
      color: #ccc;
    }

    .warning {
      border-color: #666;
      color: #999;
    }

    label {
      color: #999;
    }

    .btn-text {
      color: #999;
    }

    .btn-text:hover:not(:disabled) {
      color: #f5f5f7;
    }

    .shortcut-capture {
      background: #2c2c2e;
      border-color: #666;
    }

    .shortcut-capture:focus {
      border-color: #999;
    }

    .capture-display {
      background: #3a3a3c;
      color: #f5f5f7;
    }

    .shortcut-display {
      background: #2c2c2e;
      border-color: #3a3a3c;
    }

    .shortcut-value {
      color: #f5f5f7;
    }
  }
</style>
