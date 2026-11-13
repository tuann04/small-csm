"""Audio constants and helpers. CSM works on 24 kHz mono float32."""

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920   # one Mimi frame: 80 ms, i.e. 12.5 frames per second


def load_audio(path) -> np.ndarray:
    """Read an audio file as float32 mono at 24 kHz."""
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav


def soft_limit(audio: np.ndarray, knee: float = 0.8, ceiling: float = 0.98) -> np.ndarray:
    """Keep peaks inside full scale without changing the level of the rest.

    CSM output often peaks above 1.0 (up to ~1.2 seen), which clips when written to a 16-bit wav or played.
    Samples below `knee` are untouched; above it they are bent smoothly towards `ceiling`. It is one fixed
    curve, so unlike a per-chunk gain it causes no level jumps when applied to a stream chunk by chunk.
    """
    mag = np.abs(audio)
    over = mag > knee
    out = audio.copy()
    out[over] = np.sign(audio[over]) * (knee + (ceiling - knee) * np.tanh((mag[over] - knee) / (ceiling - knee)))
    return out
