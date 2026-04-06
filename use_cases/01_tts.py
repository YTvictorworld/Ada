#!/usr/bin/env python3
"""
Use case 01 - Basic TTS with Kokoro
------------------------------------
Converts text to speech using the local Kokoro model.

Usage:
    python use_cases/01_tts.py
    python use_cases/01_tts.py --text "Hola, soy Ada" --voice ef_dora
"""

import argparse
import re
import sys
from pathlib import Path

import sounddevice as sd
from kokoro_onnx import Kokoro

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = str(PROJECT_ROOT / "voices" / "kokoro-v1.0.onnx")
VOICES_PATH = str(PROJECT_ROOT / "voices" / "voices.bin")

# Available Spanish voices
# ef_dora  — Spanish female
# em_alex  — Spanish male

def speak(kokoro: Kokoro, text: str, voice: str, speed: float) -> tuple:
    """Replace 'Ada' with its Spanish phonetic form and render audio."""
    normalized = re.sub(r'\bAda\b', 'Eyda', text, flags=re.IGNORECASE)
    return kokoro.create(normalized, voice=voice, speed=speed, lang="es")


def main():
    parser = argparse.ArgumentParser(description="Basic TTS with Kokoro")
    parser.add_argument("--text", default="Hola, soy Ada. ¿En qué te puedo ayudar?")
    parser.add_argument("--voice", default="ef_dora", help="Voice to use (default: ef_dora)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed 0.5-2.0 (default: 1.0)")
    args = parser.parse_args()

    print("Loading Kokoro model...")
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

    print(f"Generating speech: '{args.text}'")
    samples, sample_rate = speak(kokoro, args.text, args.voice, args.speed)

    print(f"Playing... (sample_rate={sample_rate})")
    sd.play(samples, sample_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    main()
