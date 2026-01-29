<script lang="ts">
  import { invoke, isTauri as checkIsTauri } from "@tauri-apps/api/core";
  import { onMount } from "svelte";
  import { writable } from "svelte/store";

  interface AppState {
    isRecording: boolean;
    isTranscribing: boolean;
    lastTranscription?: string;
    error?: string;
    selectedModel?: string;
  }

  const uiState = writable({ isRecording: false, isTranscribing: false });

  onMount(() => {
    if (!checkIsTauri()) {
      return;
    }

    let intervalId: number | null = null;

    const pollState = async () => {
      try {
        const appState = await invoke<AppState>("get_app_state");
        uiState.set({
          isRecording: appState.isRecording,
          isTranscribing: appState.isTranscribing
        });
      } catch (e) {
        console.error("Failed to get state:", e);
      }
    };

    pollState();
    intervalId = setInterval(pollState, 100) as unknown as number;

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  });
</script>

<div class="indicator-container">
  <div class="indicator-card">
    {#if $uiState.isRecording}
      <div class="indicator recording">
        <div class="dot"></div>
        <span class="text">Recording</span>
      </div>
    {:else if $uiState.isTranscribing}
      <div class="indicator processing">
        <div class="spinner"></div>
        <span class="text">Processing</span>
      </div>
    {/if}
  </div>
</div>

<style>
  :global(html, body) {
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
  }

  .indicator-container {
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: transparent;
  }

  .indicator-card {
    background: rgba(30, 30, 30, 0.95);
    border-radius: 12px;
    padding: 10px 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }

  .indicator {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .text {
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 14px;
    font-weight: 500;
  }

  /* Recording dot with pulse animation */
  .dot {
    width: 12px;
    height: 12px;
    background: #ff3b30;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.6;
      transform: scale(1.15);
    }
  }

  /* Processing spinner */
  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: #007aff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  /* Light mode support */
  @media (prefers-color-scheme: light) {
    .indicator-card {
      background: rgba(255, 255, 255, 0.95);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .text {
      color: #1a1a1a;
    }

    .spinner {
      border-color: rgba(0, 0, 0, 0.2);
      border-top-color: #007aff;
    }
  }
</style>
