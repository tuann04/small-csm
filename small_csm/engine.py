"""
FastCSM: the generation engine. Most users want `CSMSpeaker` (speaker.py), which wraps this.

The Hugging Face `generate` loop is CPU-bound: per audio frame it runs one backbone step and 31 tiny
depth-decoder steps, all launched from Python. Here one frame is a single CUDA-graph replay:
backbone step -> first codebook -> 31 depth-decoder steps -> next frame's input embedding.

The file reads top to bottom as:
  1. construction     static KV caches, rotary tables, graph buffers
  2. forward passes   backbone and depth decoder (both use blocks.decoder_block)
  3. the frame graph  what one replay does, and capturing it
  4. prompt, context  voice prefix and conversation turns, kept as a KV snapshot
  5. generation       stream(): restore the snapshot, prefill the text, replay frames, decode audio

Weights are the Hugging Face `CsmForConditionalGeneration` modules, used unchanged.
Design notes and measurements: docs/how-it-works.md.

Two limits to keep in mind:
- Sampling settings (temperature, depth_temperature, top_k) are baked into the captured graph
  at construction; changing the attributes later has no effect.
- One utterance at a time: the graph uses static buffers, so stream() is not re-entrant and not
  thread-safe. Wrap it in a lock if you share an instance.
"""

import logging
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .audio import FRAME_SAMPLES
from .blocks import decoder_block
from .sampling import sample

logger = logging.getLogger(__name__)


class FastCSM:
    # ------------------------------------------------------------ 1. construction

    def __init__(self, model, processor, voice_wav_24k: np.ndarray, voice_text: str,
                 temperature: float = 0.9, depth_temperature: float = 0.9, top_k: int = 50,
                 max_frames: int = 150, max_context_tokens: int = 512):
        self.model = model
        self.proc = processor
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.temperature = temperature
        self.depth_temperature = depth_temperature
        self.top_k = top_k
        self.max_frames = max_frames
        self.max_context_tokens = max_context_tokens

        cfg = model.config
        dcfg = cfg.depth_decoder_config
        self.n_cb = cfg.num_codebooks
        self.vocab = cfg.vocab_size
        self.bb_layers = model.backbone_model.layers
        self.bb_heads, self.bb_kv, self.bb_hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.dd = model.depth_decoder.model
        self.dd_heads, self.dd_kv, self.dd_hd = dcfg.num_attention_heads, dcfg.num_key_value_heads, dcfg.head_dim
        self.dd_head_w = model.depth_decoder.codebooks_head.weight   # [31, 1024, vocab]
        self.max_len = cfg.max_position_embeddings

        dev, dt = self.device, self.dtype
        self.bb_k = [torch.zeros(1, self.bb_kv, self.max_len, self.bb_hd, device=dev, dtype=dt) for _ in self.bb_layers]
        self.bb_v = [torch.zeros_like(k) for k in self.bb_k]
        self.dd_k = [torch.zeros(1, self.dd_kv, dcfg.max_position_embeddings, self.dd_hd, device=dev, dtype=dt)
                     for _ in self.dd.layers]
        self.dd_v = [torch.zeros_like(k) for k in self.dd_k]

        with torch.no_grad():
            pos = torch.arange(self.max_len, device=dev)[None]
            z = torch.zeros(1, self.max_len, 1, device=dev, dtype=dt)
            self.bb_cos, self.bb_sin = model.backbone_model.rotary_emb(z, position_ids=pos)
            pos = torch.arange(dcfg.max_position_embeddings, device=dev)[None]
            self.dd_cos, self.dd_sin = self.dd.rotary_emb(z[:, :pos.shape[1]], position_ids=pos)
            self.cb_offsets = (torch.arange(self.n_cb, device=dev) * self.vocab)

        # static graph buffers
        self.x = torch.zeros(1, 1, cfg.hidden_size, device=dev, dtype=dt)
        self.pos = torch.zeros(1, device=dev, dtype=torch.long)
        self.codes = torch.zeros(self.n_cb, device=dev, dtype=torch.long)
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        self._build_voice_prefix(voice_wav_24k, voice_text)
        # KV snapshot restored at the start of every utterance: voice prefix + conversation context
        self._turns = []   # embeddings of the context turns, oldest first
        self.snap_k, self.snap_v, self.snap_len = self.prefix_k, self.prefix_v, self.prefix_len
        self._capture()

    # ------------------------------------------------------- 2. forward passes

    def _backbone(self, x, pos_ids):
        """x [1,T,H] at positions pos_ids [T] (long tensor). Returns normed hidden [1,T,H]."""
        cos, sin = self.bb_cos[:, pos_ids], self.bb_sin[:, pos_ids]
        mask = (torch.arange(self.max_len, device=x.device)[None, :] <= pos_ids[:, None])[None, None]

        def write(kc, vc, k, v):
            kc.index_copy_(2, pos_ids, k)
            vc.index_copy_(2, pos_ids, v)

        def attend(kc, vc, q):
            return F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask, enable_gqa=True)

        for i, lyr in enumerate(self.bb_layers):
            x = decoder_block(lyr, x, cos, sin, self.bb_k[i], self.bb_v[i],
                            self.bb_heads, self.bb_kv, self.bb_hd, write, attend)
        return self.model.backbone_model.norm(x)

    def _dd_run(self, x, start):
        """Depth decoder on x [1,L,1024] at static positions start..start+L-1."""
        L = x.shape[1]
        end = start + L
        cos, sin = self.dd_cos[:, start:end], self.dd_sin[:, start:end]

        def write(kc, vc, k, v):
            kc[:, :, start:end] = k
            vc[:, :, start:end] = v

        def attend(kc, vc, q):
            return F.scaled_dot_product_attention(q, kc[:, :, :end], vc[:, :, :end],
                                                  is_causal=L > 1, enable_gqa=True)

        for i, lyr in enumerate(self.dd.layers):
            x = decoder_block(lyr, x, cos, sin, self.dd_k[i], self.dd_v[i],
                            self.dd_heads, self.dd_kv, self.dd_hd, write, attend)
        return self.dd.norm(x)

    def _depth(self, h_last, c0):
        """h_last [1,H] backbone hidden, c0 [1] -> codes [32]."""
        proj, emb = self.dd.inputs_embeds_projector, self.dd.embed_tokens
        x = torch.cat([proj(h_last)[:, None], proj(emb(c0[:, None]))], dim=1)   # positions 0 (hidden), 1 (c0)
        codes = [c0]
        hs = self._dd_run(x, 0)[:, -1]
        c = sample(hs @ self.dd_head_w[0], self.depth_temperature, self.top_k)
        codes.append(c)
        for p in range(2, self.n_cb):
            e = proj(emb((c + (p - 1) * self.vocab)[:, None]))
            hs = self._dd_run(e, p)[:, -1]
            c = sample(hs @ self.dd_head_w[p - 1], self.depth_temperature, self.top_k)
            codes.append(c)
        return torch.cat(codes)

    # ------------------------------------------------------- 3. the frame graph

    def _frame(self):
        h = self._backbone(self.x, self.pos)[:, -1]                       # [1,H]
        c0 = sample(self.model.lm_head(h), self.temperature, self.top_k)  # [1]
        codes = self._depth(h, c0)                                         # [32]
        self.codes.copy_(codes)
        emb = self.model.backbone_model.embed_tokens.embed_audio_tokens(codes + self.cb_offsets).sum(0)
        self.x.copy_(emb.view(1, 1, -1))
        self.pos.add_(1)

    @torch.no_grad()
    def _capture(self):
        logger.info("Capturing CUDA graph...")
        self.x.zero_()
        self.pos.fill_(0)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._frame()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._frame()
        torch.cuda.synchronize()

    # ------------------------------------------- 4. voice prompt and conversation

    def _embed(self, conversation):
        inp = self.proc.apply_chat_template(conversation, tokenize=True, return_dict=True).to(self.device)
        values = inp.get("input_values")
        if values is not None:
            values = values.to(self.dtype)
        merged = self.model._merge_input_ids_with_input_values(
            inp["input_ids"], values, inp.get("input_values_cutoffs"), None)
        return merged["inputs_embeds"].to(self.dtype)

    @torch.no_grad()
    def _prefill(self, embeds, start):
        T = embeds.shape[1]
        self._backbone(embeds, torch.arange(start, start + T, device=self.device))

    @torch.no_grad()
    def _build_voice_prefix(self, wav, text):
        conv = [{"role": "0", "content": [{"type": "text", "text": text}, {"type": "audio", "path": wav}]}]
        embeds = self._embed(conv)
        self.prefix_len = embeds.shape[1]
        self._prefill(embeds, 0)
        P = self.prefix_len
        self.prefix_k = [k[:, :, :P].clone() for k in self.bb_k]
        self.prefix_v = [v[:, :, :P].clone() for v in self.bb_v]
        logger.info(f"Voice prefix cached: {P} tokens")

    @property
    def context_tokens(self) -> int:
        """Tokens of conversation context currently kept (excludes the voice prefix)."""
        return self.snap_len - self.prefix_len

    @torch.no_grad()
    def add_turn(self, text: str, audio: np.ndarray, speaker: int = 0) -> None:
        """Append a finished turn to the conversation context.

        text/audio: what was said and its 24 kHz mono float32 audio. CSM's chat template requires
        both for every context turn. speaker: the speaker id ("0" is the voice being generated).
        The oldest turns are dropped once the context exceeds max_context_tokens.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) < 4800:   # under 0.2 s: Mimi cannot encode it usefully
            return
        conv = [{"role": str(speaker), "content": [{"type": "text", "text": text},
                                                    {"type": "audio", "path": audio}]}]
        self._turns.append(self._embed(conv))
        self._rebuild_context()

    @torch.no_grad()
    def reset_context(self) -> None:
        """Forget the conversation; keep only the voice prompt."""
        self._turns = []
        self._rebuild_context()

    @torch.no_grad()
    def _rebuild_context(self):
        while self._turns and sum(t.shape[1] for t in self._turns) > self.max_context_tokens:
            self._turns.pop(0)
        P = self.prefix_len
        if not self._turns:
            self.snap_k, self.snap_v, self.snap_len = self.prefix_k, self.prefix_v, P
            return
        for i in range(len(self.bb_k)):
            self.bb_k[i][:, :, :P] = self.prefix_k[i]
            self.bb_v[i][:, :, :P] = self.prefix_v[i]
        ctx = torch.cat(self._turns, dim=1)
        self._prefill(ctx, P)
        L = P + ctx.shape[1]
        self.snap_k = [k[:, :, :L].clone() for k in self.bb_k]
        self.snap_v = [v[:, :, :L].clone() for v in self.bb_v]
        self.snap_len = L

    # ---------------------------------------------------------- 5. generation

    @torch.no_grad()
    def stream(self, text: str, first_chunk_frames: int = 4, chunk_frames: int = 6,
               context: bool = True) -> Iterator[np.ndarray]:
        """Yield float32 mono audio at 24 kHz as frames are generated.

        context=False ignores the conversation context for this utterance (voice prompt only)."""
        if context:
            P, ks, vs = self.snap_len, self.snap_k, self.snap_v
        else:
            P, ks, vs = self.prefix_len, self.prefix_k, self.prefix_v
        for i in range(len(self.bb_k)):
            self.bb_k[i][:, :, :P] = ks[i]
            self.bb_v[i][:, :, :P] = vs[i]

        embeds = self._embed([{"role": "0", "content": [{"type": "text", "text": text}]}])
        T = embeds.shape[1]
        if P + T + self.max_frames >= self.max_len:
            raise ValueError(f"text too long for the context window ({T} tokens)")
        if T > 1:
            self._prefill(embeds[:, :-1], P)
        self.x.copy_(embeds[:, -1:])
        self.pos.fill_(P + T - 1)

        # Keep the GPU busy: frame i+1 is launched before frame i is read back
        # (the graph only depends on device buffers). Frames past EOS are discarded.
        ring = 4
        pinned = torch.zeros(ring, self.n_cb, dtype=torch.long).pin_memory()
        events = [torch.cuda.Event() for _ in range(ring)]

        def launch(i):
            self.graph.replay()
            pinned[i % ring].copy_(self.codes, non_blocking=True)
            events[i % ring].record()

        frames, emitted, next_emit = [], 0, first_chunk_frames
        launch(0)
        for i in range(self.max_frames):
            if i + 1 < self.max_frames:
                launch(i + 1)
            events[i % ring].synchronize()
            codes = pinned[i % ring].clone()
            if bool((codes[:-1] == 0).all()):
                break
            frames.append(codes)
            if len(frames) >= next_emit:
                audio = self._decode(frames)
                yield audio[emitted * FRAME_SAMPLES:]
                emitted = len(frames)
                next_emit = emitted + chunk_frames
        if len(frames) > emitted:
            audio = self._decode(frames)
            yield audio[emitted * FRAME_SAMPLES:]

    @torch.no_grad()
    def _decode(self, frames) -> np.ndarray:
        codes = torch.stack(frames).to(self.device)            # [F, 32]
        out = self.model.codec_model.decode(codes.transpose(0, 1).unsqueeze(0))
        return out.audio_values[0, 0].float().cpu().numpy()
