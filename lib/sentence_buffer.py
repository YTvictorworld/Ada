"""Accumulate streaming tokens and emit complete sentences."""

import re

# Match sentence boundary: ., !, ? possibly followed by closing quote/paren
SENTENCE_END = re.compile(r'([.!?]+["\')\]]?)(\s|$)')

# Common abbreviations that contain a period but don't end sentences
ABBREV = {"sr.", "sra.", "dr.", "dra.", "lic.", "ing.", "etc.", "ej.", "vs."}


class SentenceBuffer:
    """Buffers streaming text and yields complete sentences.

    Sentences are emitted only when they reach min_chars to avoid
    flushing fragments like "Sí." that are too short for natural TTS.
    """

    def __init__(self, min_chars: int = 15):
        self._buf = ""
        self.min_chars = min_chars

    def feed(self, chunk: str) -> list[str]:
        """Add a chunk and return any complete sentences ready to speak."""
        self._buf += chunk
        sentences = []

        while True:
            sentence, rest = self._extract_one()
            if sentence is None:
                break
            sentences.append(sentence)
            self._buf = rest

        return sentences

    def _extract_one(self) -> tuple[str | None, str]:
        """Try to extract one complete sentence from the buffer."""
        for match in SENTENCE_END.finditer(self._buf):
            end_idx = match.end(1)
            candidate = self._buf[:end_idx].strip()

            # Check if too short
            if len(candidate) < self.min_chars:
                continue

            # Check if it ends with an abbreviation
            last_word = candidate.split()[-1].lower() if candidate.split() else ""
            if last_word in ABBREV:
                continue

            return candidate, self._buf[end_idx:].lstrip()

        return None, self._buf

    def flush(self) -> str | None:
        """Return any remaining text after the stream ends."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None
