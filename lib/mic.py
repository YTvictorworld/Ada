"""Cross-platform microphone detection and resolution."""

import platform

import numpy as np


def get_input_devices() -> list[tuple[int, str, str]]:
    """List all available input (microphone) devices.

    Returns:
        List of (index, name, api_name) tuples.
    """
    import pyaudio

    pa = pyaudio.PyAudio()
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            api = pa.get_host_api_info_by_index(info["hostApi"])["name"]
            devices.append((i, info["name"], api))
    pa.terminate()
    return devices


def resolve_mic_index(name_query: str) -> int:
    """Find a microphone by partial name match.

    Uses platform-specific API preference:
        - Windows: MME > DirectSound > WASAPI
        - Linux: ALSA > PulseAudio > JACK
        - macOS/other: no preference, first match

    Args:
        name_query: Partial name to match (case-insensitive).

    Returns:
        Device index for PyAudio.

    Raises:
        ValueError: If no microphone matches the query.
    """
    matches = [(i, name, api) for i, name, api in get_input_devices()
               if name_query.lower() in name.lower()]

    if not matches:
        raise ValueError(f"No microphone found matching '{name_query}'")

    system = platform.system()
    if system == "Windows":
        api_preference = ["MME", "DirectSound", "WASAPI"]
    elif system == "Linux":
        api_preference = ["ALSA", "PulseAudio", "JACK"]
    else:
        api_preference = []

    for pref in api_preference:
        preferred = [m for m in matches if pref in m[2]]
        if preferred:
            return preferred[0][0]

    return matches[0][0]


def list_mics():
    """Print all available microphones to stdout."""
    devices = get_input_devices()
    if not devices:
        print("No microphones found.")
        return
    print("Available microphones:")
    for i, name, api in devices:
        print(f"  {i}: {name} [{api}]")


def test_mic(device_index: int | None = None, duration: float = 2.0) -> bool:
    """Record a short sample and check if the microphone captures audio.

    Args:
        device_index: PyAudio device index, or None for system default.
        duration: Seconds to record.

    Returns:
        True if audio was detected, False if silent/broken.
    """
    import pyaudio

    pa = pyaudio.PyAudio()

    if device_index is not None:
        info = pa.get_device_info_by_index(device_index)
    else:
        info = pa.get_default_input_device_info()

    rate = int(info["defaultSampleRate"])
    chunk = 1024

    print(f"[mic] Testing microphone for {duration:.0f}s... (say something)")

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk,
        )
    except Exception as e:
        print(f"[mic] ERROR: could not open microphone: {e}")
        pa.terminate()
        return False

    frames = []
    num_chunks = int(rate / chunk * duration)
    for _ in range(num_chunks):
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        except Exception:
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if not frames:
        print("[mic] ERROR: no audio data captured.")
        return False

    samples = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2)))
    peak = float(np.max(np.abs(samples)))

    if rms < 0.005:
        print(f"[mic] WARNING: microphone seems silent (RMS={rms:.4f}, Peak={peak:.4f})")
        print("[mic] Check that the microphone is connected and not muted.")
        return False

    if peak >= 0.99:
        print(f"[mic] WARNING: audio is clipping (Peak={peak:.4f}). Lower the mic volume.")

    print(f"[mic] Microphone OK (RMS={rms:.4f}, Peak={peak:.4f})")
    return True
