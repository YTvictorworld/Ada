#!/usr/bin/env python3
"""
Use case 04 - Ada: full conversational pipeline
------------------------------------------------
STT (RealtimeSTT) -> Router (wake word) -> LLM (llama-server, streaming) -> TTS (Kokoro)

Requires llama-server running externally:
    llama-server -m /path/to/model.gguf --port 8080

Usage:
    python use_cases/04_ada.py
    python use_cases/04_ada.py --mic "USB PnP"
    python use_cases/04_ada.py --endpoint http://localhost:8080
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import load_config
from lib.device import ensure_silero_vad, ensure_whisper_model, patch_torch_hub
from lib.llm import make_llm
from lib.mic import get_input_devices, list_mics, resolve_mic_index, test_mic
from lib.router import ConversationRouter
from lib.sentence_buffer import SentenceBuffer
from lib.tts import KokoroTTS

patch_torch_hub()

from RealtimeSTT import AudioToTextRecorder

# Sentinel to stop the TTS player thread
_STOP = object()


def tts_player_thread(tts: KokoroTTS, q: queue.Queue, recorder_ref: dict, on_done=None):
    """Background thread that consumes sentences and speaks them in order.

    Mutes the recorder while speaking so Ada does not transcribe her own voice.
    """
    speaking = False
    while True:
        item = q.get()
        if item is _STOP:
            break

        recorder = recorder_ref.get("recorder")
        if not speaking and recorder is not None:
            recorder.set_microphone(False)
            speaking = True

        try:
            tts.speak(item)
        except Exception as e:
            print(f"\n[tts error] {e}")

        # Re-enable mic only when the queue is fully drained.
        if speaking and q.empty() and recorder is not None:
            try:
                recorder.clear_audio_queue()
            except Exception:
                pass
            recorder.set_microphone(True)
            speaking = False
            if on_done is not None:
                try:
                    on_done()
                except Exception as e:
                    print(f"\n[router error] {e}")


def main():
    config = load_config()
    stt_cfg = config.get("stt", {})
    llm_cfg = config.get("llm", {})

    parser = argparse.ArgumentParser(description="Ada conversational assistant")
    parser.add_argument("--wake", default="ada", help="Wake word (default: ada)")
    parser.add_argument("--model", default=None, help="Whisper model")
    parser.add_argument("--lang", default=None, help="Language code")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--mic", default=None, help="Partial microphone name")
    parser.add_argument("--endpoint", default=None, help="llama-server endpoint")
    parser.add_argument("--list-mics", action="store_true", help="List microphones and exit")
    args = parser.parse_args()

    if args.list_mics:
        list_mics()
        return

    model = args.model or config["model"]
    language = args.lang or config["language"]
    device = args.device or config["device"]
    compute_type = config["compute_type"]
    mic_name = args.mic or config.get("microphone")

    # Resolve LLM provider and apply --endpoint override only when it makes sense.
    provider = llm_cfg.get("provider", "llama-server")
    if args.endpoint:
        if provider == "llama-server":
            llm_cfg = {**llm_cfg, "llama_server": {**(llm_cfg.get("llama_server") or {}), "endpoint": args.endpoint}}
        else:
            print(f"[warn] --endpoint ignored when provider={provider}")

    mic_index = resolve_mic_index(mic_name) if mic_name else None
    if mic_index is not None:
        _, name, api = [(i, n, a) for i, n, a in get_input_devices() if i == mic_index][0]
        print(f"[mic] {name} [{api}] (index {mic_index})")

    if not test_mic(mic_index, duration=1.5):
        print("[mic] Microphone test failed. Continuing anyway...\n")

    print(f"[config] model={model} | device={device} | compute={compute_type} | lang={language}")
    print(f"[llm] provider={provider} | history_size={llm_cfg.get('history_size', 20)}")
    print()

    # Pre-checks
    print("[init] Loading TTS (Kokoro)...")
    try:
        tts = KokoroTTS(voice="ef_dora")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)

    print(f"[init] Initializing LLM backend ({provider})...")
    try:
        llm = make_llm(llm_cfg)
    except (ImportError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)
    err = llm.check_connection()
    if err:
        print(f"[error] {err}")
        sys.exit(1)
    print(f"[init] {provider} OK.")

    ensure_whisper_model(model)
    realtime_model = stt_cfg.get("realtime_model_type", "tiny")
    if realtime_model != model:
        ensure_whisper_model(realtime_model)
    ensure_silero_vad()
    print()

    # TTS player thread
    tts_queue: queue.Queue = queue.Queue()
    recorder_ref: dict = {"recorder": None, "router": None}

    def _on_tts_done():
        r = recorder_ref.get("router")
        if r is not None:
            r.start_follow_up()

    player = threading.Thread(
        target=tts_player_thread,
        args=(tts, tts_queue, recorder_ref, _on_tts_done),
        daemon=True,
    )
    player.start()

    def dispatch(command: str):
        """Called by the router when a command for Ada is detected."""
        ts = time.strftime("%H:%M:%S")
        print(f"\r\033[92m[{ts}] [USER] {command}\033[0m")
        print("\033[95m[ADA] \033[0m", end="", flush=True)

        buffer = SentenceBuffer(min_chars=15)
        try:
            for chunk in llm.stream(command):
                print(f"\033[95m{chunk}\033[0m", end="", flush=True)
                for sentence in buffer.feed(chunk):
                    tts_queue.put(sentence)
        except Exception as e:
            print(f"\n[llm error] {e}")
            return

        # Flush remaining
        leftover = buffer.flush()
        if leftover:
            tts_queue.put(leftover)
        print()

    router = ConversationRouter(
        on_dispatch=dispatch,
        wake_word=args.wake,
        max_wait=10.0,
        min_words=1,
        follow_up_window=10.0,
    )
    recorder_ref["router"] = router

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
        initial_prompt=stt_cfg.get("initial_prompt", ""),
        beam_size=stt_cfg.get("beam_size", 1),
        beam_size_realtime=stt_cfg.get("beam_size_realtime", 1),
        realtime_processing_pause=stt_cfg.get("realtime_processing_pause", 0.2),
        post_speech_silence_duration=stt_cfg.get("post_speech_silence_duration", 0.4),
        min_length_of_recording=stt_cfg.get("min_length_of_recording", 0.2),
        silero_sensitivity=stt_cfg.get("silero_sensitivity", 0.4),
        enable_realtime_transcription=True,
        realtime_model_type=realtime_model,
        no_log_file=True,
        early_transcription_on_silence=stt_cfg.get("early_transcription_on_silence", 0.2),
        use_main_model_for_realtime=False,
        batch_size=16,
    )
    if mic_index is not None:
        recorder_kwargs["input_device_index"] = mic_index

    recorder = AudioToTextRecorder(**recorder_kwargs)
    recorder_ref["recorder"] = recorder

    print(f"Ada is ready. Say '{args.wake}' followed by a command. (Ctrl+C to stop)\n")

    try:
        while True:
            recorder.text(on_final_text)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tts_queue.put(_STOP)
        player.join(timeout=2)
        recorder.shutdown()


if __name__ == "__main__":
    main()
