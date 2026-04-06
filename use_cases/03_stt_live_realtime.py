#!/usr/bin/env python3
"""
Use case 03 - Real-time STT with RealtimeSTT
----------------------------------------------
Captures microphone audio and transcribes in real time
using RealtimeSTT (faster-whisper + built-in VAD).

No external server required. Runs entirely locally.

Usage:
    python use_cases/03_stt_live_realtime.py
    python use_cases/03_stt_live_realtime.py --model medium --lang es
    python use_cases/03_stt_live_realtime.py --mic "USB PnP"
    python use_cases/03_stt_live_realtime.py --list-mics
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

patch_torch_hub()

from RealtimeSTT import AudioToTextRecorder


def main():
    config = load_config()
    stt = config.get("stt", {})

    parser = argparse.ArgumentParser(description="Real-time STT with RealtimeSTT")
    parser.add_argument("--model", default=None, help="Whisper model: tiny|base|small|medium|large-v3")
    parser.add_argument("--lang", default=None, help="Language for transcription")
    parser.add_argument("--device", default=None, help="Device: cuda|cpu")
    parser.add_argument("--mic", default=None, help="Partial microphone name (e.g. 'USB', 'Realtek')")
    parser.add_argument("--list-mics", action="store_true", help="List available microphones and exit")
    args = parser.parse_args()

    if args.list_mics:
        list_mics()
        return

    # CLI args override config
    model = args.model or config["model"]
    language = args.lang or config["language"]
    device = args.device or config["device"]
    compute_type = config["compute_type"]
    mic_name = args.mic or config.get("microphone")

    mic_index = resolve_mic_index(mic_name) if mic_name else None
    if mic_index is not None:
        _, name, api = [(i, n, a) for i, n, a in get_input_devices() if i == mic_index][0]
        print(f"[mic] {name} [{api}] (index {mic_index})")

    # Quick mic test
    if not test_mic(mic_index, duration=1.5):
        print("[mic] Microphone test failed. Use 'python main.py test-mic' for diagnostics.")
        print("[mic] Continuing anyway...\n")

    print(f"[config] model={model} | device={device} | compute={compute_type} | lang={language}")
    print()

    # Check if models are cached or need downloading
    ensure_whisper_model(model)
    realtime_model = stt.get("realtime_model_type", "tiny")
    if realtime_model != model:
        ensure_whisper_model(realtime_model)
    ensure_silero_vad()
    print()

    speech_end_time = [0.0]

    def on_recording_stop():
        speech_end_time[0] = time.perf_counter()

    def on_realtime_update(text: str):
        if text.strip():
            print(f"\r\033[90m  ... {text}\033[0m", end="", flush=True)

    def on_final_text(text: str):
        if text.strip():
            elapsed = time.perf_counter() - speech_end_time[0] if speech_end_time[0] else 0
            print(f"\r\033[96m  >> {text}  \033[93m({elapsed:.2f}s)\033[0m")

    recorder_kwargs = dict(
        model=model,
        language=language,
        device=device,
        compute_type=compute_type,
        on_realtime_transcription_update=on_realtime_update,
        on_recording_stop=on_recording_stop,
        spinner=False,
        initial_prompt=stt.get("initial_prompt", ""),
        beam_size=stt.get("beam_size", 1),
        beam_size_realtime=stt.get("beam_size_realtime", 1),
        realtime_processing_pause=stt.get("realtime_processing_pause", 0.2),
        post_speech_silence_duration=stt.get("post_speech_silence_duration", 0.4),
        min_length_of_recording=stt.get("min_length_of_recording", 0.2),
        silero_sensitivity=stt.get("silero_sensitivity", 0.4),
        enable_realtime_transcription=True,
        realtime_model_type=stt.get("realtime_model_type", "tiny"),
        no_log_file=True,
        early_transcription_on_silence=stt.get("early_transcription_on_silence", 0.2),
        use_main_model_for_realtime=False,
        batch_size=16,
    )
    if mic_index is not None:
        recorder_kwargs["input_device_index"] = mic_index

    recorder = AudioToTextRecorder(**recorder_kwargs)

    print("Microphone active. Speak... (Ctrl+C to stop)\n")

    try:
        while True:
            recorder.text(on_final_text)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        recorder.shutdown()


if __name__ == "__main__":
    main()
