"""Minimal CLI for small-csm."""

import argparse
import time
from pathlib import Path

import soundfile as sf

from .audio import SAMPLE_RATE
from .speaker import CSMSpeaker


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize speech with Sesame CSM-1B (fp16, CUDA graph).")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="out.wav", help="Output WAV path (default: out.wav)")
    parser.add_argument("--voice", default=None, help="Voice name in voices/ or path to a .wav file")
    parser.add_argument("--max-frames", type=int, default=150, help="Max frames to generate (12.5 frames = 1s)")
    args = parser.parse_args()

    spk = CSMSpeaker(voice=args.voice)
    t0 = time.perf_counter()
    first_chunk_ms = None
    chunks = []
    for chunk in spk.stream(args.text, max_frames=args.max_frames):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - t0) * 1000
        chunks.append(chunk)
    total_s = time.perf_counter() - t0

    if not chunks:
        print("No audio generated.")
        return

    import numpy as np

    audio = np.concatenate(chunks)
    dur_s = len(audio) / SAMPLE_RATE
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, SAMPLE_RATE)
    rtf = total_s / dur_s if dur_s > 0 else 0.0
    print(f"Saved {out_path} ({dur_s:.2f}s audio | first chunk {first_chunk_ms:.0f} ms | total {total_s:.2f}s | RTF {rtf:.2f})")


if __name__ == "__main__":
    main()
