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
    FOLLOW_UP = "follow_up"

    # Phonetic variants Whisper commonly produces for "Ada" (pronounced "Eyda").
    DEFAULT_ALIASES = {
        "ada": ["ada", "eida", "eyda", "aida", "hada", "heida", "aída", "eda", "ida"],
    }

    def __init__(
        self,
        on_dispatch: Callable[[str], None],
        wake_word: str = "ada",
        max_wait: float = 10.0,
        min_words: int = 2,
        aliases: Optional[list[str]] = None,
        follow_up_window: float = 10.0,
    ):
        self.on_dispatch = on_dispatch
        self.wake_word = wake_word.lower()
        # Build the full list of accepted wake-word forms.
        forms = aliases if aliases is not None else self.DEFAULT_ALIASES.get(self.wake_word, [])
        self.wake_forms = sorted({self.wake_word, *(f.lower() for f in forms)}, key=len, reverse=True)
        self._wake_re = re.compile(
            r'\b(' + '|'.join(re.escape(f) for f in self.wake_forms) + r')\b',
            re.IGNORECASE,
        )
        self.max_wait = max_wait
        self.min_words = min_words
        self.follow_up_window = follow_up_window

        self._state = self.LISTENING
        self._capture_started_at: Optional[float] = None
        self._follow_up_until: Optional[float] = None

    @property
    def state(self) -> str:
        return self._state

    def on_partial(self, text: str) -> None:
        """Hook for RealtimeSTT's on_realtime_transcription_update.

        Detects wake word as early as possible to enter CAPTURING state.
        In FOLLOW_UP mode, any speech transitions straight to CAPTURING.
        """
        if self._state == self.FOLLOW_UP:
            if self._follow_up_until and time.perf_counter() > self._follow_up_until:
                self._reset()
                return
            if text and text.strip():
                self._state = self.CAPTURING
                self._capture_started_at = time.perf_counter()
            return

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
        # Expire follow-up window if nothing was captured in time
        if self._state == self.FOLLOW_UP:
            if self._follow_up_until and time.perf_counter() > self._follow_up_until:
                self._reset()
                return

        # Timeout from CAPTURING without a final → reset
        if self._state == self.CAPTURING and self._capture_started_at is not None:
            elapsed = time.perf_counter() - self._capture_started_at
            if elapsed > self.max_wait:
                self._reset()
                return

        came_from_follow_up = self._state in (self.FOLLOW_UP, self.CAPTURING) and self._follow_up_until is not None

        # In follow-up mode, the wake word is not required.
        if not came_from_follow_up and not self._contains_wake_word(text):
            self._reset()
            return

        if came_from_follow_up and self._contains_wake_word(text):
            command = self._extract_command(text)
        elif came_from_follow_up:
            command = text.strip()
        else:
            command = self._extract_command(text)

        if not command or len(command.split()) < self.min_words:
            self._reset()
            return

        self._state = self.DISPATCHING
        try:
            self.on_dispatch(command)
        finally:
            # Stay in DISPATCHING; ada.py calls start_follow_up() once TTS ends.
            self._capture_started_at = None

    def start_follow_up(self) -> None:
        """Open a follow-up window where the wake word is not required."""
        self._state = self.FOLLOW_UP
        self._capture_started_at = None
        self._follow_up_until = time.perf_counter() + self.follow_up_window

    def _contains_wake_word(self, text: str) -> bool:
        if not text:
            return False
        return self._wake_re.search(text) is not None

    def _extract_command(self, text: str) -> str:
        """Extract everything after the wake word, removing leading punctuation."""
        match = self._wake_re.search(text)
        if not match:
            return text.strip()
        after = text[match.end():]
        return after.lstrip(" ,.:;!?¡¿").strip()

    def _reset(self) -> None:
        self._state = self.LISTENING
        self._capture_started_at = None
        self._follow_up_until = None
