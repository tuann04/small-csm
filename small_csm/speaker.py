"""
CSMSpeaker: load CSM-1B in fp16 and speak with a fixed voice prompt.

    spk = CSMSpeaker()
    for chunk in spk.stream("Hello there"):   # float32 mono, 24 kHz, as it is generated
        play(chunk)
    audio = spk.synthesize("Hello there")      # whole utterance as one array

The default voice is the Expresso prompt ex03_happy in tts/csm/voices/expresso/. A voice prompt is
a short clip (about 10-15 s) of the target voice plus its exact transcript
(<name>.txt next to the wav, or voice_text=...).

Conversation context (add_turn / reset_context) makes the next line sound like a reply in that conversation.
See docs/usage.md.
"""

import logging
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from .audio import SAMPLE_RATE, load_audio
from .engine import FastCSM

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sesame/csm-1b"
DEFAULT_VOICE = Path(__file__).parent / "voices" / "expresso" / "ex03_happy.wav"


def load_voice(wav_path, text: Optional[str] = None):
    """Read a voice prompt wav (resampled to 24 kHz mono) and its transcript."""
    wav_path = Path(wav_path)
    wav = load_audio(wav_path)
    if text is None:
        txt = wav_path.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"voice transcript not found: {txt} (pass voice_text=...)")
        text = txt.read_text().strip()
    return wav, text


class CSMSpeaker:
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        voice_wav: Optional[os.PathLike] = None,
        voice_text: Optional[str] = None,
        temperature: float = 0.9,
        depth_temperature: float = 0.9,
        top_k: int = 50,
        max_frames: int = 150,
        max_context_tokens: int = 512,
        warmup: bool = True,
    ):
        """
        temperature / depth_temperature: sampling for the first codebook / the other 31
            (<= 0 is greedy). Fixed at construction (baked into the CUDA graph).
        max_frames: longest utterance in frames (12.5 frames = 1 s). Prefix + text + max_frames
            must fit the 2048-token context.
        max_context_tokens: budget for conversation context (see add_turn); oldest turns are dropped first.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required (CUDA graphs)")
        from transformers import AutoProcessor, CsmForConditionalGeneration

        start = time.time()
        wav, text = load_voice(voice_wav or DEFAULT_VOICE, voice_text)
        processor = AutoProcessor.from_pretrained(model_id)
        model = CsmForConditionalGeneration.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
        model.eval()
        self._fast = FastCSM(model, processor, wav, text, temperature=temperature,
                             depth_temperature=depth_temperature, top_k=top_k, max_frames=max_frames,
                             max_context_tokens=max_context_tokens)
        if warmup:
            for _ in self._fast.stream("hello, how are you"):
                pass
        logger.info(f"CSM ready in {time.time() - start:.1f}s")

    def stream(self, text: str, first_chunk_frames: int = 6, chunk_frames: int = 8,
               context: bool = True) -> Iterator[np.ndarray]:
        """Yield float32 mono audio at 24 kHz. The first chunk is first_chunk_frames long
        (80 ms per frame), later chunks chunk_frames. context=False ignores the conversation
        context for this utterance. Not re-entrant."""
        yield from self._fast.stream(text, first_chunk_frames, chunk_frames, context)

    def add_turn(self, text: str, audio: np.ndarray, speaker: int = 0) -> None:
        """Add a finished turn (its text and 24 kHz mono audio) to the conversation context, so the next
        utterance is spoken in that conversational setting. speaker 0 is the voice being generated;
        use another id (for example 1) for the person talking to it. Both text and audio are required."""
        self._fast.add_turn(text, audio, speaker)

    def reset_context(self) -> None:
        """Forget the conversation (keep the voice prompt). Call between unrelated conversations."""
        self._fast.reset_context()

    @property
    def context_tokens(self) -> int:
        return self._fast.context_tokens

    def synthesize(self, text: str, context: bool = True) -> np.ndarray:
        parts = list(self.stream(text, context=context))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)
