# Ada Assistant POC

Proof of concept for a fully local voice assistant: Speech-to-Text + LLM + Text-to-Speech.

**Pipeline**: Microphone -> RealtimeSTT -> Wake-word router -> llama-server (LLM) -> Kokoro TTS -> Speakers

**Target platform**: Linux (primary). Also runs on Windows for development.

## Prerequisites

- Python 3.10+
- ffmpeg
- PortAudio (for audio capture)

### Installing PortAudio

**Linux (Debian/Ubuntu):**
```bash
sudo apt install portaudio19-dev ffmpeg
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install portaudio-devel ffmpeg
```

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Windows:**
PortAudio is bundled with PyAudio. You only need ffmpeg:
```bash
choco install ffmpeg
```

## Installation

### Quick setup (recommended)

```bash
pip install -r requirements.txt
python main.py setup
```

The interactive wizard will:
1. Install missing dependencies
2. Auto-detect your GPU/CUDA
3. Ask which Whisper model to use (with description and VRAM for each)
4. Ask for the language
5. List available microphones and let you choose
6. Generate `config.yaml` with your configuration

If you run any command without having done setup, it launches automatically.

### Manual setup

If you prefer to configure manually:

**Install PyTorch with NVIDIA GPU:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Without GPU (CPU only):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**TTS models** (only if using TTS): download `kokoro-v1.0.onnx` and `voices.bin` to `voices/` directory.

## Configuration

On first run, a `config.yaml` file is automatically generated from `config.example.yaml`.

```yaml
microphone: null       # Partial mic name or null for default
model: base            # tiny | base | small | medium | large-v3
language: es           # Language code
device: auto           # cuda | cpu | auto
compute_type: auto     # float16 | int8 | auto
```

CLI arguments always take priority over the config file.

### GPU auto-detection

When `device` or `compute_type` are set to `auto`, the system:
1. Checks if `torch` has CUDA support
2. If an NVIDIA GPU is available: uses `cuda` + `float16`
3. Otherwise: uses `cpu` + `int8`

## Usage

`main.py` is the central entry point for the project:

```bash
python main.py setup              # Interactive setup (first run)
python main.py list-mics          # List available microphones
python main.py test-mic           # Test microphone audio capture
python main.py check-gpu          # Check GPU/CUDA status
python main.py check-models       # Check downloaded models status
python main.py stt                # Launch real-time STT
python main.py router             # Launch STT + wake word router (no LLM/TTS)
python main.py ada                # Launch full conversational assistant
python main.py tts "Hello world"  # Generate speech from text
```

### STT - Real-time Speech-to-Text

```bash
# Use a specific microphone
python main.py stt --mic "USB PnP"

# Override model and language
python main.py stt --model medium --lang es --mic "USB PnP"

# Use a specific device
python main.py stt --device cuda
```

The transcription output shows:
- **Gray** (`...`): partial real-time transcription (tiny model, visual feedback only)
- **Cyan** (`>>`): final transcription (configured model, this is the actual result)
- **Yellow** (`Xs`): transcription latency

### Router - STT + Wake Word

Real-time STT that only dispatches sentences containing the wake word ("Ada"). Useful as a building block for the full pipeline.

```bash
python main.py router --mic "USB PnP"
python main.py router --wake "ada" --mic "USB PnP"
```

Output legend:
- **Gray** `... text`: partial transcription (LISTENING)
- **Gray** `* ... text`: wake word detected (CAPTURING)
- **Cyan** `>> text`: final transcription
- **Green** `[DISPATCH] -> text`: command extracted and dispatched

### Ada - Full conversational assistant

The full pipeline: STT -> Router (wake word) -> LLM (llama-server, SSE streaming) -> TTS (Kokoro).

**Requires**: `llama-server` running externally with any GGUF model. Ada checks the connection on startup and aborts with a clear message if it can't reach the server.

```bash
# Terminal 1: start llama-server with any GGUF model
llama-server -m /path/to/model.gguf --port 8080

# Terminal 2: launch Ada
python main.py ada --mic "USB PnP"
```

Ada listens continuously. Say "Ada" followed by your command:
- "Oye Ada, qué hora es"
- "Ada, cuéntame un chiste"
- "Ada, qué te dije antes" (uses sliding window history)

LLM configuration in `config.yaml`:
```yaml
llm:
  endpoint: "http://localhost:8080"      # llama-server URL
  history_size: 20                       # sliding window of past messages
  temperature: 0.7
  system_prompt: >-
    Eres Ada, una asistente de voz femenina en espanol...
```

**How it works**:
1. **STT** transcribes everything in real-time (RealtimeSTT + Silero VAD)
2. **Router** filters by wake word "Ada", discards everything else
3. **LLM** receives the command + sliding window history; streams response token-by-token via SSE
4. **SentenceBuffer** accumulates tokens and emits complete sentences as soon as `.`, `!`, or `?` arrives
5. **TTS player thread** consumes sentences from a queue and plays them with Kokoro (back-to-back, no waiting)

This means the first sentence starts playing while the LLM is still generating the rest — feels much more responsive than a batch pipeline.

**CLI overrides**:
```bash
python main.py ada --endpoint http://192.168.1.50:8080  # remote llama-server
python main.py ada --wake "computadora"                  # custom wake word
python main.py ada --model small                         # smaller STT model
```

### TTS - Text-to-Speech

```bash
python main.py tts "Hola, soy Ada"
python main.py tts "Buenos dias" --voice em_alex --speed 1.2
```

Available voices: `ef_dora` (female), `em_alex` (male).

### Utilities

```bash
# List microphones
python main.py list-mics

# Test microphone
python main.py test-mic --mic "USB PnP"

# Check GPU
python main.py check-gpu

# Check downloaded models
python main.py check-models
python main.py check-models --model large-v3
```

## Microphone

You can configure the microphone in three ways:
1. **config.yaml**: `microphone: "USB PnP"`
2. **CLI**: `--mic "USB PnP"`
3. **null/omitted**: uses the system default microphone

Audio API selection adapts to the operating system:
- **Windows**: prefers MME > DirectSound > WASAPI
- **Linux**: prefers ALSA > PulseAudio > JACK

## Troubleshooting

### PyAudio won't install on Linux
```bash
sudo apt install portaudio19-dev python3-dev
pip install pyaudio
```

### CUDA not detected
Verify that torch has CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If it says `False`, reinstall torch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Microphone not detected
1. Use `--list-mics` to see available devices
2. Set the name in `config.yaml` or use `--mic`
3. On Linux, verify PulseAudio/ALSA are working: `arecord -l`

### Slow transcription
- Use a smaller model: `--model base` or `--model small`
- Verify CUDA is active (not CPU)
- STT parameters can be adjusted in `config.yaml` under the `stt` section

### Ada can't connect to llama-server
```
[error] Cannot connect to llama-server at http://localhost:8080. Is it running?
```
Make sure `llama-server` is running and listening on the configured port:
```bash
llama-server -m /path/to/model.gguf --port 8080
```
Verify manually:
```bash
curl http://localhost:8080/health
```
If your `llama-server` is on another machine, set the endpoint in `config.yaml` or pass `--endpoint`:
```bash
python main.py ada --endpoint http://192.168.1.50:8080
```

### Ada doesn't speak
- Make sure Kokoro model files exist in `voices/` (run `python main.py setup` to download)
- Check that the LLM is actually returning content (you should see magenta tokens in the console)
- If you see tokens but no audio, verify your output device is working

## Architecture

```
              ┌──────────┐
   Mic ────►  │ RealtimeSTT │ ──┐
              └──────────┘   │
                             ▼
                       ┌──────────┐
                       │  Router  │  filters by wake word "Ada"
                       └──────────┘
                             │ command
                             ▼
                       ┌──────────┐      SSE      ┌──────────────┐
                       │   LLM    │ ◄───────────► │ llama-server │
                       │  client  │               │  (external)  │
                       └──────────┘               └──────────────┘
                             │ token stream
                             ▼
                    ┌────────────────┐
                    │ SentenceBuffer │  splits into sentences
                    └────────────────┘
                             │ sentences
                             ▼
                       ┌──────────┐
                       │  Queue   │
                       └──────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ TTS thread     │  Kokoro
                    │ (Kokoro)       │  ef_dora voice
                    └────────────────┘
                             │
                             ▼
                         Speakers
```

## Project structure

```
whisper-poc/
├── main.py                  # CLI entry point
├── config.example.yaml      # Config template
├── lib/
│   ├── config.py           # Config loader + setup wizard
│   ├── device.py           # GPU/CUDA detection, model downloads
│   ├── mic.py              # Microphone enumeration + test
│   ├── router.py           # Wake-word router (LISTENING/CAPTURING/DISPATCHING)
│   ├── llm.py              # llama-server HTTP client (SSE streaming)
│   ├── sentence_buffer.py  # Token stream → sentences
│   └── tts.py              # Kokoro TTS wrapper
├── use_cases/
│   ├── 01_tts.py           # Standalone TTS demo
│   ├── 02_router.py        # STT + router (no LLM)
│   ├── 03_stt_live_realtime.py  # Standalone STT
│   └── 04_ada.py           # Full pipeline
└── voices/
    ├── kokoro-v1.0.onnx    # Kokoro model (gitignored)
    └── voices.bin          # Kokoro voices (gitignored)
```
