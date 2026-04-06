"""Conversation router: filters STT output by wake word before dispatching."""

import re
import time
from typing import Callable, Optional


class ConversationRouter:
    """Routes STT transcriptions to a callback only when the wake word is detected.

    Hooks into RealtimeSTT callbacks:
        - on_partial: connected to `on_realtime_transcription_update`
        - on_final:   connected to `recorder.text()` callback
    """

    LISTENING = "listening"
    CAPTURING = "capturing"
    DISPATCHING = "dispatching"

    def __init__(
        self,
        on_dispatch: Callable[[str], None],
        wake_word: str = "ada",
        max_wait: float = 10.0,
        min_words: int = 2,
    ):
        self.on_dispatch = on_dispatch
        self.wake_word = wake_word.lower()
        self.max_wait = max_wait
        self.min_words = min_words

        self._state = self.LISTENING
        self._capture_started_at: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def on_partial(self, text: str) -> None:
        """Hook for RealtimeSTT's on_realtime_transcription_update.

        Detects wake word as early as possible to enter CAPTURING state.
        """
        if self._state != self.LISTENING:
            return

        if self._contains_wake_word(text):
            self._state = self.CAPTURING
            self._capture_started_at = time.perf_counter()

    def on_final(self, text: str) -> None:
        """Hook for the callback passed to recorder.text().

        Receives the full sentence when RealtimeSTT detects end of speech.
        Decides whether to dispatch or discard.
        """
        # Timeout from CAPTURING without a final → reset
        if self._state == self.CAPTURING and self._capture_started_at is not None:
            elapsed = time.perf_counter() - self._capture_started_at
            if elapsed > self.max_wait:
                self._reset()
                return

        if not self._contains_wake_word(text):
            # Not addressed to Ada → discard
            self._reset()
            return

        command = self._extract_command(text)
        if not command or len(command.split()) < self.min_words:
            self._reset()
            return

        self._state = self.DISPATCHING
        try:
            self.on_dispatch(command)
        finally:
            self._reset()

    def _contains_wake_word(self, text: str) -> bool:
        if not text:
            return False
        return self.wake_word in text.lower()

    def _extract_command(self, text: str) -> str:
        """Extract everything after the wake word, removing leading punctuation."""
        pattern = re.compile(rf'\b{re.escape(self.wake_word)}\b', re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return text.strip()
        after = text[match.end():]
        return after.lstrip(" ,.:;!?¡¿").strip()

    def _reset(self) -> None:
        self._state = self.LISTENING
        self._capture_started_at = None
