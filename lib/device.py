"""GPU/CUDA auto-detection, torch configuration, and model management."""

import platform
from pathlib import Path


def patch_torch_hub():
    """Bypass torch.hub untrusted repo validation for silero-vad.
    Needed on all platforms due to GitHub API rate limits."""
    try:
        import torch
        torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None
    except ImportError:
        pass


def detect_device() -> tuple[str, str]:
    """Auto-detect the best device and compute type.

    Returns:
        (device, compute_type): e.g. ("cuda", "float16") or ("cpu", "int8")
    """
    try:
        import torch
    except ImportError:
        print("[device] torch not installed, using CPU + int8")
        return ("cpu", "int8")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print("[device] Using cuda + float16")
        return ("cuda", "float16")

    print("[device] No CUDA GPU detected, using CPU + int8")
    if platform.system() == "Linux":
        print("[device] Hint: install CUDA toolkit + torch with CUDA for GPU acceleration:")
        print("[device]   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124")

    return ("cpu", "int8")


def check_whisper_model(model: str) -> bool:
    """Check if a whisper model is already downloaded. Logs status.

    Returns:
        True if model is cached locally, False if it needs downloading.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
        repo_id = f"Systran/faster-whisper-{model}"
        result = try_to_load_from_cache(repo_id, "config.json")
        if result is not None and isinstance(result, (str, Path)):
            print(f"[model] '{model}' found in cache: {Path(result).parent}")
            return True
    except (ImportError, Exception):
        pass

    print(f"[model] '{model}' not found locally, will download from Hugging Face (Systran/faster-whisper-{model})")
    return False


def ensure_whisper_model(model: str):
    """Ensure a whisper model is downloaded. Downloads if not cached."""
    if check_whisper_model(model):
        return

    print(f"[model] Downloading '{model}'... (this may take a while)")
    try:
        from huggingface_hub import snapshot_download
        repo_id = f"Systran/faster-whisper-{model}"
        path = snapshot_download(repo_id)
        print(f"[model] '{model}' downloaded to: {path}")
    except Exception as e:
        print(f"[model] ERROR downloading '{model}': {e}")
        print(f"[model] The model will be downloaded automatically when RealtimeSTT starts.")


def check_silero_vad() -> bool:
    """Check if Silero VAD model is cached locally.

    Returns:
        True if cached, False if it needs downloading.
    """
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master"
    if cache_dir.exists():
        print(f"[model] Silero VAD found in cache: {cache_dir}")
        return True

    print("[model] Silero VAD not found locally, will download from GitHub (snakers4/silero-vad)")
    return False


def ensure_silero_vad():
    """Ensure Silero VAD is downloaded."""
    if check_silero_vad():
        return

    print("[model] Downloading Silero VAD...")
    try:
        patch_torch_hub()
        import torch
        torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        print("[model] Silero VAD downloaded.")
    except Exception as e:
        print(f"[model] ERROR downloading Silero VAD: {e}")
        print("[model] It will be downloaded automatically when RealtimeSTT starts.")
