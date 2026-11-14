# Small CSM

`small-csm` packages Sesame `CSM-1B` in `float16` for Python developers running speech synthesis on an NVIDIA GPU. Hugging Face's default `generate` loop launches one backbone step and 31 depth-decoder steps separately from Python for every audio frame. This package captures the frame pipeline into one CUDA graph and holds the voice prompt in a cached KV snapshot. It streams 24 kHz mono `float32` audio through a lightweight wrapper around the original Hugging Face model weights.

## Requirements

- Python 3.10 or newer (`requires-python >= 3.10`)
- NVIDIA GPU with CUDA support (`CSMSpeaker` raises `RuntimeError` if CUDA is unavailable)
- Hugging Face authentication with access accepted for the gated `sesame/csm-1b` repository

The original package notes estimate roughly 3.7 GB of VRAM for `float16` inference, though exact memory usage depends on `max_frames` and conversation context length.

## Installation

Install the package from the repository root with `pip` or `uv`:

```bash
pip install .
# or with uv
uv pip install .
```

Installation pulls in `numpy`, `soundfile`, `torch`, `torchaudio`, `transformers`, and `accelerate`, and registers the `small-csm` CLI entry point.

## Quick Start

For Python applications, instantiate `CSMSpeaker` once and stream chunks as frames arrive or synthesize a full clip in one call:

```python
from small_csm import CSMSpeaker, soft_limit

spk = CSMSpeaker()

# Stream float32 mono chunks at 24 kHz as frames are generated
for chunk in spk.stream("Hey, how is it going?"):
    bounded = soft_limit(chunk)
    ...

# Synthesize a complete utterance as one NumPy array
audio = spk.synthesize("Wait, seriously? Congratulations!")
```

You can also invoke the installed command-line script for quick single-utterance synthesis to a WAV file:

```bash
small-csm "Hey, how is it going?" -o out.wav
```

## Voice Prompts and Conversation Context

By default, `CSMSpeaker` loads `small_csm/voices/expresso/ex03_happy.wav` along with its matching `.txt` transcript. To speak in a custom voice, pass a 10 to 15 second reference clip to `voice_wav` and supply its transcript either through `voice_text` or as a sibling `<name>.txt` file. `load_audio` and `load_voice` automatically downmix stereo clips to mono and resample them to 24 kHz.

`CSM-1B` conditions its prosody on prior dialogue turns. Call `add_turn(text, audio, speaker=0)` after an utterance finishes to append text and 24 kHz mono audio into the conversation history. Use `speaker=0` for the generated voice and another integer, such as `1`, for an interlocutor. `FastCSM` skips clips shorter than 4,800 samples (0.2 seconds) because the Mimi codec cannot encode them reliably. When stored turns exceed `max_context_tokens` (default `512`), `FastCSM` drops the oldest turns first and rebuilds the KV snapshot from the voice prefix. Pass `context=False` to `stream()` or `synthesize()` to ignore conversation history for a single utterance, or call `reset_context()` to clear all turns while keeping the voice prompt.

CSM output frequently peaks above `1.0` (up to around `1.2` in observed runs), which causes clipping in 16-bit WAV files and audio playback. Calling `soft_limit(audio, knee=0.8, ceiling=0.98)` bends peaks smoothly above `knee` using a fixed `tanh` curve. Because the curve stays fixed across calls, applying it chunk by chunk during streaming introduces no volume jumps.

## How CUDA-Graph Generation Works

`FastCSM` (`small_csm/engine.py`) keeps the `CsmForConditionalGeneration` backbone and depth-decoder weights untouched while replacing the Python loop with static GPU buffers. Both transformers execute through a shared `decoder_block` helper (`small_csm/blocks.py`) that applies RMSNorm, grouped-query attention with rotary embeddings, and a gated MLP, differing only in injected KV-cache callbacks. During construction, `FastCSM` allocates fixed-size KV caches (`2048` backbone positions), prefills the voice prompt once into a saved snapshot, and captures `_frame()` inside a `torch.cuda.CUDAGraph`.

Each graph replay runs one backbone step, samples codebook 0 (`c0`) via `sample()` (`small_csm/sampling.py`), and executes all 31 depth-decoder steps. It then sums the resulting audio embeddings for the next step and increments the position tensor. During `stream()`, the engine restores the KV snapshot and prefills the input text. Next, it overlaps GPU execution with host readback via a 4-slot pinned-memory ring buffer and CUDA events, launching frame `i + 1` before synchronizing frame `i`. `FastCSM` decodes accumulated frames (`1920` samples or `80 ms` per frame at 12.5 frames per second) into waveforms via `model.codec_model.decode`.

Two architectural limits follow directly from this static graph design:

- **Fixed sampling settings**: `temperature`, `depth_temperature`, and `top_k` are baked into the captured graph at initialization, so mutating those attributes on `FastCSM` later has no effect. Pass `temperature <= 0` or `depth_temperature <= 0` at construction time for greedy decoding.
- **Single-utterance concurrency**: Because `_frame()` reads and writes shared static device buffers, `stream()` is neither re-entrant nor thread-safe. Guard shared `CSMSpeaker` instances with a lock when serving concurrent requests.

## Python API Reference

The `small_csm` package exports `CSMSpeaker`, `FastCSM`, `SAMPLE_RATE` (`24000`), `load_audio`, `load_voice`, and `soft_limit`.

| Parameter | Default | Role |
| :--- | :--- | :--- |
| `model_id` | `"sesame/csm-1b"` | Hugging Face model identifier |
| `voice_wav` | `ex03_happy.wav` | Path to reference voice WAV |
| `voice_text` | `None` | Transcript if no `.txt` exists |
| `temperature` | `0.9` | Codebook 0 sampling temperature |
| `depth_temperature` | `0.9` | Codebooks 1 to 31 temperature |
| `top_k` | `50` | Top-k filter for sampling |
| `max_frames` | `150` | Frame limit (12.5 frames/s) |
| `max_context_tokens` | `512` | Conversation turn token budget |
| `warmup` | `True` | Run one warmup utterance |

The `CSMSpeaker.stream()` method accepts `first_chunk_frames=6` (`480 ms`), `chunk_frames=8` (`640 ms`), and `context=True`. Prefix tokens, input text tokens, and `max_frames` together must stay below the `2048`-token backbone context window.

## Bundled Voices and License

Eight reference voice prompts sit in `small_csm/voices/expresso/`, covering speakers `ex01` and `ex03` (male) and `ex02` and `ex04` (female) in `default` and `happy` styles. Each clip joins four to seven read sentences from the Expresso benchmark (`ylacombe/expresso`, Nguyen et al., 2023), trimmed, resampled to 24 kHz, and peak-normalized alongside a matching `.txt` transcript.

Those bundled voice files carry a **CC BY-NC 4.0** license for non-commercial use with attribution, as documented in `small_csm/voices/expresso/LICENSE.md`. Replace them with your own `voice_wav` and `voice_text` prompts when deploying commercial applications.

