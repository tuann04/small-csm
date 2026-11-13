"""Token sampling used inside the captured CUDA graph."""

import torch


def sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Pick one token id per row. logits [1, V] -> ids [1].

    temperature <= 0 is greedy. Otherwise top-k, softmax at `temperature`, then multinomial. All of it
    is graph-capturable (no host synchronization; the RNG is PyTorch's graph-safe generator).
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    vals, idx = torch.topk(logits.float() / temperature, top_k, dim=-1)
    pick = torch.multinomial(torch.softmax(vals, dim=-1), 1)
    return idx.gather(-1, pick).squeeze(-1)
