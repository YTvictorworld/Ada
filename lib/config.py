"""Configuration loading, first-run setup, and interactive wizard."""

import subprocess
import sys
from pathlib import Path

import yaml

from lib.device import detect_device
from lib.mic import get_input_devices

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
EXAMPLE_PATH = PROJECT_ROOT / "config.example.yaml"

MODELS = [
    ("tiny",     "~1GB VRAM",  "Very fast, low accuracy. Good for quick tests."),
    ("base",     "~1GB VRAM",  "Fast, acceptable accuracy. Good starting point."),
    ("small",    "~2GB VRAM",  "Good speed/accuracy balance. Recommended for live STT."),
    ("medium",   "~5GB VRAM",  "Slow but very accurate. Good with a powerful GPU."),
    ("large-v3", "~10GB VRAM", "Most accurate. Requires GPU with lots of VRAM."),
]

LANGUAGES = [
    ("es", "Spanish"),
    ("en", "English"),
    ("fr", "French"),
    ("pt", "Portuguese"),
    ("de", "German"),
]


def config_exists() -> bool:
    """Check if config.yaml exists."""
    return CONFIG_PATH.exists()


def load_config() -> dict:
    """Load config.yaml. Must exist (run setup first).

    Auto-resolves 'auto' values for device and compute_type.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Resolve auto device/compute_type
    if config.get("device") == "auto" or config.get("compute_type") == "auto":
        device, compute_type = detect_device()
        if config.get("device") == "auto":
            config["device"] = device
        if config.get("compute_type") == "auto":
            config["compute_type"] = compute_type

    return config


def run_setup():
    """Interactive setup wizard. Creates config.yaml step by step."""
    print()
    print("=" * 55)
    print("  Whisper POC - Setup")
    print("=" * 55)
    print()

    # Step 1: Install dependencies
    _step_dependencies()

    # Step 2: Detect GPU / install torch
    device, compute_type = _step_gpu()

    # Step 3: TTS models (optional)
    _step_tts_models()

    # Step 4: Choose model
    model = _step_model(device)

    # Step 5: Choose language
    language = _step_language()

    # Step 6: Choose microphone
    microphone = _step_microphone()

    # Step 7: Write config
    _write_config(model, language, device, compute_type, microphone)

    print()
    print("=" * 55)
    print("  Setup complete!")
    print("=" * 55)
    print(f"  Config saved to: {CONFIG_PATH}")
    print(f"  Edit config.yaml to adjust advanced settings.")
    print()


def _prompt_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """Show numbered options and return the selected index."""
    while True:
        try:
            raw = input(f"\n{prompt} [{default + 1}]: ").strip()
            if not raw:
                return default
            choice = int(raw) - 1
            if 0 <= choice < len(options):
                return choice
            print(f"  Choose between 1 and {len(options)}")
        except ValueError:
            print(f"  Enter a number between 1 and {len(options)}")
        except (KeyboardInterrupt, EOFError):
            print("\nSetup cancelled.")
            sys.exit(1)


def _step_dependencies():
    """Check and install pip dependencies."""
    print("[1/6] Checking dependencies...")
    print()

    # Core packages to check (name_to_import: pip_package)
    CORE_PACKAGES = {
        "yaml": "pyyaml",
        "numpy": "numpy",
        "soundfile": "soundfile",
        "pyaudio": "pyaudio",
        "faster_whisper": "faster-whisper",
        "RealtimeSTT": "RealtimeSTT",
    }

    OPTIONAL_PACKAGES = {
        "kokoro_onnx": "kokoro_onnx",
        "sounddevice": "sounddevice",
    }

    missing_core = []
    for import_name, pip_name in CORE_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"  [OK] {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name}")
            missing_core.append(pip_name)

    missing_optional = []
    for import_name, pip_name in OPTIONAL_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"  [OK] {pip_name} (optional)")
        except ImportError:
            print(f"  [MISSING] {pip_name} (optional, needed for TTS)")
            missing_optional.append(pip_name)

    all_missing = missing_core + missing_optional

    if not all_missing:
        print("\n  All dependencies installed.")
        return

    print(f"\n  Missing packages: {', '.join(all_missing)}")
    try:
        answer = input("\n  Install missing packages? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.")
        sys.exit(1)

    if answer in ("", "y", "yes"):
        print()
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + all_missing + ["-q"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  Error installing packages:\n{result.stderr}")
        else:
            print("  Packages installed successfully.")
    else:
        print("  Skipped. Install manually: pip install " + " ".join(all_missing))


def _step_gpu() -> tuple[str, str]:
    """Detect GPU and offer to install appropriate torch."""
    print()
    print("[2/6] Detecting GPU...")
    print()

    # Check if torch is installed at all
    torch_installed = False
    cuda_available = False
    try:
        import torch
        torch_installed = True
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    if not torch_installed:
        print("  [MISSING] torch / torchaudio")
        print()
        print("  PyTorch is required but not installed.")
        print("  Choose an install option:\n")
        print("    1) With CUDA (NVIDIA GPU, recommended if you have one)")
        print("    2) CPU only (no GPU acceleration, slower)")
        print("    3) Skip (install manually later)")

        try:
            choice = input("\n  Option [1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSetup cancelled.")
            sys.exit(1)

        if choice in ("", "1"):
            print("\n  Installing torch with CUDA support...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu124", "-q"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  Error installing torch:\n{result.stderr}")
                return ("cpu", "int8")
            print("  torch + CUDA installed successfully.")
            return ("cuda", "float16")
        elif choice == "2":
            print("\n  Installing torch (CPU only)...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  Error installing torch:\n{result.stderr}")
            else:
                print("  torch (CPU) installed successfully.")
            return ("cpu", "int8")
        else:
            print("  Skipped. Install manually before running STT.")
            return ("cpu", "int8")

    # torch is installed, detect device
    device, compute_type = detect_device()

    if not cuda_available:
        print()
        print("  torch is installed but CUDA is not available.")
        print("  If you have an NVIDIA GPU, reinstall torch with CUDA:")
        print("    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
        print()

        try:
            answer = input("  Reinstall torch with CUDA now? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nSetup cancelled.")
            sys.exit(1)

        if answer in ("y", "yes"):
            print("\n  Reinstalling torch with CUDA...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cu124",
                 "--force-reinstall", "-q"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
            else:
                print("  torch + CUDA reinstalled. Restart setup to detect GPU.")
                return ("cuda", "float16")

    return device, compute_type


def _step_tts_models():
    """Offer to download Kokoro TTS model files."""
    print()
    print("[3/6] TTS models (optional)...")
    print()

    voices_dir = PROJECT_ROOT / "voices"
    voices_dir.mkdir(exist_ok=True)
    model_path = voices_dir / "kokoro-v1.0.onnx"
    voices_path = voices_dir / "voices.bin"

    model_ok = model_path.exists()
    voices_ok = voices_path.exists()

    if model_ok and voices_ok:
        print(f"  [OK] kokoro-v1.0.onnx ({model_path.stat().st_size / 1024**2:.0f} MB)")
        print(f"  [OK] voices.bin ({voices_path.stat().st_size / 1024**2:.0f} MB)")
        print("  TTS models already downloaded.")
        return

    if not model_ok:
        print("  [MISSING] kokoro-v1.0.onnx (~170 MB)")
    else:
        print(f"  [OK] kokoro-v1.0.onnx")
    if not voices_ok:
        print("  [MISSING] voices.bin (~27 MB)")
    else:
        print(f"  [OK] voices.bin")

    print()
    print("  TTS models are needed for text-to-speech (use case 01).")
    print("  Source: github.com/thewh1teagle/kokoro-onnx")

    try:
        answer = input("\n  Download TTS models now? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.")
        sys.exit(1)

    if answer not in ("", "y", "yes"):
        print("  Skipped. Download manually if you need TTS.")
        return

    BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

    for filename, filepath in [("kokoro-v1.0.onnx", model_path), ("voices-v1.0.bin", voices_path)]:
        if filepath.exists() and filename == "kokoro-v1.0.onnx":
            continue
        if filepath.exists() and filename == "voices-v1.0.bin":
            continue

        url = f"{BASE_URL}/{filename}"
        target = filepath if filename != "voices-v1.0.bin" else voices_path
        print(f"\n  Downloading {filename}...")

        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(target), _download_progress)
            print(f"\n  [OK] Saved to {target}")
        except Exception as e:
            print(f"\n  Error downloading {filename}: {e}")
            print(f"  Download manually from: {url}")


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / 1024**2
        mb_total = total_size / 1024**2
        print(f"\r  {percent:.0f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def _step_model(device: str) -> str:
    """Let user choose a Whisper model."""
    print()
    print("[4/6] Select Whisper model:")
    print()

    for i, (name, vram, desc) in enumerate(MODELS):
        marker = ""
        if device == "cpu" and name in ("medium", "large-v3"):
            marker = " (not recommended without GPU)"
        print(f"  {i + 1}) {name:<10} {vram:<12} {desc}{marker}")

    default = 2 if device == "cuda" else 1  # small for GPU, base for CPU
    choice = _prompt_choice("Model", [m[0] for m in MODELS], default)
    selected = MODELS[choice][0]
    print(f"  -> {selected}")
    return selected


def _step_language() -> str:
    """Let user choose language."""
    print()
    print("[5/6] Select language:")
    print()

    for i, (code, name) in enumerate(LANGUAGES):
        print(f"  {i + 1}) {code} - {name}")

    choice = _prompt_choice("Language", [l[0] for l in LANGUAGES], 0)
    selected = LANGUAGES[choice][0]
    print(f"  -> {selected}")
    return selected


def _step_microphone() -> str | None:
    """Let user choose a microphone."""
    print()
    print("[6/6] Select microphone:")
    print()

    devices = get_input_devices()
    if not devices:
        print("  No microphones found. Using system default.")
        return None

    # Deduplicate by name (same mic appears under different APIs)
    seen = {}
    unique_devices = []
    for idx, name, api in devices:
        if name not in seen:
            seen[name] = (idx, name, api)
            unique_devices.append((idx, name, api))

    print(f"  1) System default")
    for i, (idx, name, api) in enumerate(unique_devices):
        print(f"  {i + 2}) {name} [{api}]")

    choice = _prompt_choice("Microphone", ["default"] + [d[1] for d in unique_devices], 0)

    if choice == 0:
        print("  -> System default")
        return None

    selected = unique_devices[choice - 1][1]
    print(f"  -> {selected}")
    return selected


def _write_config(model: str, language: str, device: str, compute_type: str, microphone: str | None):
    """Write config.yaml from example template with user choices applied."""
    # Load example as base if it exists
    if EXAMPLE_PATH.exists():
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Apply user choices
    config["microphone"] = microphone
    config["model"] = model
    config["language"] = language
    config["device"] = device
    config["compute_type"] = compute_type

    # Write with comments from example template
    if EXAMPLE_PATH.exists():
        with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
            template = f.read()

        # Replace values in template
        import re
        template = re.sub(r'^(microphone:).*$', f'microphone: {_yaml_value(microphone)}', template, flags=re.MULTILINE)
        template = re.sub(r'^(model:).*$', f'model: {model}', template, flags=re.MULTILINE)
        template = re.sub(r'^(language:).*$', f'language: {language}', template, flags=re.MULTILINE)
        template = re.sub(r'^(device:).*$', f'device: {device}', template, flags=re.MULTILINE)
        template = re.sub(r'^(compute_type:).*$', f'compute_type: {compute_type}', template, flags=re.MULTILINE)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(template)
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _yaml_value(val) -> str:
    """Format a Python value for inline YAML."""
    if val is None:
        return "null"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)
