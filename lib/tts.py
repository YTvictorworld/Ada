"""Reusable Kokoro TTS wrapper."""

import re
from pathlib import Path

import sounddevice as sd

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "voices" / "kokoro-v1.0.onnx"
VOICES_PATH = PROJECT_ROOT / "voices" / "voices.bin"

# Phonetic fixes for words Kokoro mispronounces in Spanish
PHONETIC_FIXES = {
    r'\bAda\b': 'Eyda',
}


def normalize_text(text: str) -> str:
    """Apply phonetic fixes to improve Kokoro pronunciation."""
    for pattern, replacement in PHONETIC_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


class KokoroTTS:
    """Synchronous Kokoro TTS wrapper. Loads model once, reuses for many calls."""

    def __init__(self, voice: str = "ef_dora", speed: float = 1.0, lang: str = "es"):
        if not MODEL_PATH.exists() or not VOICES_PATH.exists():
            raise FileNotFoundError(
                f"Kokoro model files not found in {MODEL_PATH.parent}.\n"
                "Run 'python main.py setup' to download them."
            )
        from kokoro_onnx import Kokoro
        self.kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        self.voice = voice
        self.speed = speed
        self.lang = lang

    def render(self, text: str) -> tuple:
        """Generate audio samples for text. Returns (samples, sample_rate)."""
        normalized = normalize_text(text)
        return self.kokoro.create(
            normalized, voice=self.voice, speed=self.speed, lang=self.lang
        )

    def speak(self, text: str) -> None:
        """Render and play text. Blocks until playback finishes."""
        samples, sample_rate = self.render(text)
        sd.play(samples, sample_rate)
        sd.wait()
