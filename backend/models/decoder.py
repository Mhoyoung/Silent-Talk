"""CTC 디코더. MVP에서는 greedy(best-path) 디코딩만 제공한다.

추후 prefix-beam 디코더 + KenLM 한국어 언어모델로 확장 가능.
"""

from __future__ import annotations

import torch


class GreedyCTCDecoder:
    """argmax + collapse-repeats + remove-blank."""

    def __init__(self, blank_id: int = 0) -> None:
        self.blank_id = blank_id

    def __call__(self, logits: torch.Tensor) -> list[list[int]]:
        """logits: (T, B, V) → batch별 토큰 id 시퀀스."""
        ids = logits.argmax(dim=-1).transpose(0, 1)  # (B, T)
        results: list[list[int]] = []
        for row in ids.tolist():
            collapsed: list[int] = []
            prev = -1
            for token in row:
                if token != prev and token != self.blank_id:
                    collapsed.append(token)
                prev = token
            results.append(collapsed)
        return results
