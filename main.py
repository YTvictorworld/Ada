#!/usr/bin/env python3
"""
Whisper POC - Central CLI
-------------------------
Entry point for all project commands.

Usage:
    python main.py setup              Interactive setup (first run)
    python main.py list-mics          List available microphones
    python main.py check-gpu          Check GPU/CUDA status
    python main.py check-models       Check downloaded models status
    python main.py stt                Launch real-time STT
    python main.py tts "Hello world"  Generate speech from text
"""

import argparse
import importlib
import sys

from lib.config import config_exists, load_config, run_setup
from lib.device import check_silero_vad, check_whisper_model, detect_device, patch_torch_hub
from lib.mic import list_mics, resolve_mic_index, test_mic


def require_config():
    """Ensure config.yaml exists. If not, run setup."""
    if not config_exists():
        print("config.yaml not found. Starting setup...\n")
        run_setup()


def cmd_setup(_args):
    """Run interactive setup."""
    run_setup()


def cmd_list_mics(_args):
    """List available microphones."""
    list_mics()


def cmd_test_mic(args):
    """Test a microphone."""
    mic_name = args.mic
    if mic_name:
        mic_index = resolve_mic_index(mic_name)
    elif config_exists():
        config = load_config()
        mic_name = config.get("microphone")
        mic_index = resolve_mic_index(mic_name) if mic_name else None
    else:
        mic_index = None

    test_mic(mic_index, duration=3.0)


def cmd_check_gpu(_args):
    """Check GPU/CUDA status."""
    device, compute_type = detect_device()
    print(f"\nResult: device={device}, compute_type={compute_type}")


def cmd_check_models(args):
    """Check if required models are downloaded."""
    config = load_config()
    model = args.model or config["model"]
    stt = config.get("stt", {})
    realtime_model = stt.get("realtime_model_type", "tiny")

    print("Checking models...\n")
    check_whisper_model(model)
    if realtime_model != model:
        check_whisper_model(realtime_model)
    check_silero_vad()


def cmd_stt(args):
    """Launch real-time STT."""
    patch_torch_hub()

    stt_argv = []
    if args.mic:
        stt_argv.extend(["--mic", args.mic])
    if args.model:
        stt_argv.extend(["--model", args.model])
    if args.lang:
        stt_argv.extend(["--lang", args.lang])
    if args.device:
        stt_argv.extend(["--device", args.device])

    sys.argv = ["03_stt_live_realtime.py"] + stt_argv
    mod = importlib.import_module("use_cases.03_stt_live_realtime")
    mod.main()


def cmd_ada(args):
    """Launch Ada: full STT -> Router -> LLM -> TTS pipeline."""
    patch_torch_hub()

    ada_argv = []
    if args.wake:
        ada_argv.extend(["--wake", args.wake])
    if args.mic:
        ada_argv.extend(["--mic", args.mic])
    if args.model:
        ada_argv.extend(["--model", args.model])
    if args.lang:
        ada_argv.extend(["--lang", args.lang])
    if args.device:
        ada_argv.extend(["--device", args.device])
    if args.endpoint:
        ada_argv.extend(["--endpoint", args.endpoint])

    sys.argv = ["04_ada.py"] + ada_argv
    mod = importlib.import_module("use_cases.04_ada")
    mod.main()


def cmd_router(args):
    """Launch conversation router (STT + wake word filter)."""
    patch_torch_hub()

    router_argv = []
    if args.wake:
        router_argv.extend(["--wake", args.wake])
    if args.mic:
        router_argv.extend(["--mic", args.mic])
    if args.model:
        router_argv.extend(["--model", args.model])
    if args.lang:
        router_argv.extend(["--lang", args.lang])
    if args.device:
        router_argv.extend(["--device", args.device])

    sys.argv = ["02_router.py"] + router_argv
    mod = importlib.import_module("use_cases.02_router")
    mod.main()


def cmd_tts(args):
    """Launch TTS (Kokoro - fast, lightweight)."""
    tts_argv = []
    if args.text:
        tts_argv.extend(["--text", args.text])
    if args.voice:
        tts_argv.extend(["--voice", args.voice])
    if args.speed:
        tts_argv.extend(["--speed", str(args.speed)])

    sys.argv = ["01_tts.py"] + tts_argv
    mod = importlib.import_module("use_cases.01_tts")
    mod.main()


def main():
    parser = argparse.ArgumentParser(
        description="Whisper POC - CLI central",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup           Interactive setup (first run)
  list-mics       List available microphones
  test-mic        Test microphone audio capture
  check-gpu       Check GPU/CUDA status
  check-models    Check if Whisper models are downloaded
  stt             Launch real-time speech-to-text
  router          Launch STT + wake word router
  ada             Launch Ada conversational assistant (STT+LLM+TTS)
  tts             Generate speech from text
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # setup
    subparsers.add_parser("setup", help="Interactive setup wizard")

    # list-mics
    subparsers.add_parser("list-mics", help="List available microphones")

    # test-mic
    p_test_mic = subparsers.add_parser("test-mic", help="Test microphone audio capture")
    p_test_mic.add_argument("--mic", default=None, help="Microphone name to test")

    # check-gpu
    subparsers.add_parser("check-gpu", help="Check GPU/CUDA status")

    # check-models
    p_models = subparsers.add_parser("check-models", help="Check if models are downloaded")
    p_models.add_argument("--model", default=None, help="Model to check (uses config default)")

    # stt
    p_stt = subparsers.add_parser("stt", help="Launch real-time speech-to-text")
    p_stt.add_argument("--mic", default=None, help="Microphone name")
    p_stt.add_argument("--model", default=None, help="Whisper model")
    p_stt.add_argument("--lang", default=None, help="Language code")
    p_stt.add_argument("--device", default=None, help="cuda or cpu")

    # router
    p_router = subparsers.add_parser("router", help="Launch STT + wake word router")
    p_router.add_argument("--wake", default=None, help="Wake word (default: ada)")
    p_router.add_argument("--mic", default=None, help="Microphone name")
    p_router.add_argument("--model", default=None, help="Whisper model")
    p_router.add_argument("--lang", default=None, help="Language code")
    p_router.add_argument("--device", default=None, help="cuda or cpu")

    # ada (full pipeline)
    p_ada = subparsers.add_parser("ada", help="Launch Ada conversational assistant")
    p_ada.add_argument("--wake", default=None, help="Wake word (default: ada)")
    p_ada.add_argument("--mic", default=None, help="Microphone name")
    p_ada.add_argument("--model", default=None, help="Whisper model")
    p_ada.add_argument("--lang", default=None, help="Language code")
    p_ada.add_argument("--device", default=None, help="cuda or cpu")
    p_ada.add_argument("--endpoint", default=None, help="llama-server endpoint URL")

    # tts
    p_tts = subparsers.add_parser("tts", help="Generate speech from text")
    p_tts.add_argument("text", nargs="?", default=None, help="Text to speak")
    p_tts.add_argument("--voice", default=None, help="Voice: ef_dora, em_alex")
    p_tts.add_argument("--speed", type=float, default=None, help="Speed 0.5-2.0")

    args = parser.parse_args()

    # Commands that don't require config
    NO_CONFIG_COMMANDS = {"setup", "list-mics", "test-mic", "check-gpu"}

    commands = {
        "setup": cmd_setup,
        "list-mics": cmd_list_mics,
        "test-mic": cmd_test_mic,
        "check-gpu": cmd_check_gpu,
        "check-models": cmd_check_models,
        "stt": cmd_stt,
        "router": cmd_router,
        "ada": cmd_ada,
        "tts": cmd_tts,
    }

    if args.command is None:
        if not config_exists():
            run_setup()
        else:
            parser.print_help()
        return

    # Require config for commands that need it
    if args.command not in NO_CONFIG_COMMANDS:
        require_config()

    if args.command in commands:
        commands[args.command](args)


if __name__ == "__main__":
    main()
