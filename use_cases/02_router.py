#!/usr/bin/env python3
"""
Use case 02 - Conversation Router
----------------------------------
Real-time STT + wake-word router. Only sentences containing the wake word
("Ada" by default) are dispatched to a handler (LLM, TTS, etc).

For now, the dispatch handler just prints the extracted command.

Usage:
    python use_cases/02_router.py
    python use_cases/02_router.py --wake "ada"
    python use_cases/02_router.py --mic "USB PnP"
"""

import argparse
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import load_config
from lib.device import ensure_silero_vad, ensure_whisper_model, patch_torch_hub
from lib.mic import get_input_devices, list_mics, resolve_mic_index, test_mic
from lib.router import ConversationRouter

patch_torch_hub()

from RealtimeSTT import AudioToTextRecorder


def main():
    config = load_config()
    stt = config.get("stt", {})

    parser = argparse.ArgumentParser(description="Conversation Router with wake word")
    parser.add_argument("--wake", default="ada", help="Wake word (default: ada)")
    parser.add_argument("--model", default=None, help="Whisper model")
    parser.add_argument("--lang", default=None, help="Language code")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--mic", default=None, help="Partial microphone name")
    parser.add_argument("--list-mics", action="store_true", help="List microphones and exit")
    parser.add_argument("--max-wait", type=float, default=10.0,
                        help="Max wait after wake word before discarding (default: 10s)")
    parser.add_argument("--min-words", type=int, default=2,
                        help="Min words after wake word to dispatch (default: 2)")
    args = parser.parse_args()

    if args.list_mics:
        list_mics()
        return

    model = args.model or config["model"]
    language = args.lang or config["language"]
    device = args.device or config["device"]
    compute_type = config["compute_type"]
    mic_name = args.mic or config.get("microphone")

    mic_index = resolve_mic_index(mic_name) if mic_name else None
    if mic_index is not None:
        _, name, api = [(i, n, a) for i, n, a in get_input_devices() if i == mic_index][0]
        print(f"[mic] {name} [{api}] (index {mic_index})")

    if not test_mic(mic_index, duration=1.5):
        print("[mic] Microphone test failed. Continuing anyway...\n")

    print(f"[config] model={model} | device={device} | compute={compute_type} | lang={language}")
    print(f"[router] wake_word='{args.wake}' | min_words={args.min_words} | max_wait={args.max_wait}s")
    print()

    ensure_whisper_model(model)
    realtime_model = stt.get("realtime_model_type", "tiny")
    if realtime_model != model:
        ensure_whisper_model(realtime_model)
    ensure_silero_vad()
    print()

    # Dispatch handler — for now just log to stdout
    def dispatch(command: str):
        ts = time.strftime("%H:%M:%S")
        print(f"\r\033[92m[{ts}] [DISPATCH] -> {command}\033[0m")

    router = ConversationRouter(
        on_dispatch=dispatch,
        wake_word=args.wake,
        max_wait=args.max_wait,
        min_words=args.min_words,
    )

    def on_realtime_update(text: str):
        if text.strip():
            state_marker = "*" if router.state == router.CAPTURING else " "
            print(f"\r\033[90m {state_marker} ... {text}\033[0m", end="", flush=True)
        router.on_partial(text)

    def on_final_text(text: str):
        if text.strip():
            print(f"\r\033[96m  >> {text}\033[0m")
        router.on_final(text)

    recorder_kwargs = dict(
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        on_realtime_transcription_update=on_realtime_update,
        spinner=False,
        initial_prompt=stt.get("initial_prompt", ""),
        beam_size=stt.get("beam_size", 1),
        beam_size_realtime=stt.get("beam_size_realtime", 1),
        realtime_processing_pause=stt.get("realtime_processing_pause", 0.2),
        post_speech_silence_duration=stt.get("post_speech_silence_duration", 0.4),
        min_length_of_recording=stt.get("min_length_of_recording", 0.2),
        silero_sensitivity=stt.get("silero_sensitivity", 0.4),
        enable_realtime_transcription=True,
        realtime_model_type=realtime_model,
        no_log_file=True,
        early_transcription_on_silence=stt.get("early_transcription_on_silence", 0.2),
        use_main_model_for_realtime=False,
        batch_size=16,
    )
    if mic_index is not None:
        recorder_kwargs["input_device_index"] = mic_index

    recorder = AudioToTextRecorder(**recorder_kwargs)

    print(f"Router active. Say '{args.wake}' followed by a command. (Ctrl+C to stop)\n")

    try:
        while True:
            recorder.text(on_final_text)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        recorder.shutdown()


if __name__ == "__main__":
    main()
