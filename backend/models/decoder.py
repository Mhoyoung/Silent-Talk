"""CTC 디코더 + 신뢰도(confidence) 계산.

MVP는 greedy(best-path) 디코더. 확장: prefix-beam + 한국어 LM.

confidence : 0~1 정규화값 (시퀀스 평균 softmax score)
raw_score  : CTC log-prob 원본 합산 (음수)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecodedSequence:
    ids: list[int]
    confidence: float        # 0~1
    raw_score: float         # CTC log-prob 합 (음수)


class GreedyCTCDecoder:
    """argmax + collapse-repeats + remove-blank + 신뢰도 산출."""

    def __init__(self, blank_id: int = 0) -> None:
        self.blank_id = blank_id

    def __call__(self, logits: torch.Tensor) -> list[list[int]]:
        """후방 호환: logits (T, B, V) → batch별 token id 시퀀스."""
        return [d.ids for d in self.decode(logits)]

    def decode(self, logits: torch.Tensor) -> list[DecodedSequence]:
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        top_p, top_i = probs.max(dim=-1)         # (T, B), (T, B)
        top_p = top_p.transpose(0, 1)            # (B, T)
        top_i = top_i.transpose(0, 1)
        top_logp = log_probs.gather(-1, top_i.transpose(0, 1).unsqueeze(-1)).squeeze(-1).transpose(0, 1)

        results: list[DecodedSequence] = []
        for b in range(top_i.size(0)):
            collapsed: list[int] = []
            kept_probs: list[float] = []
            kept_logp: list[float] = []
            prev = -1
            for t in range(top_i.size(1)):
                tok = int(top_i[b, t].item())
                if tok != prev and tok != self.blank_id:
                    collapsed.append(tok)
                    kept_probs.append(float(top_p[b, t].item()))
                    kept_logp.append(float(top_logp[b, t].item()))
                prev = tok
            conf = float(sum(kept_probs) / len(kept_probs)) if kept_probs else 0.0
            raw = float(sum(kept_logp)) if kept_logp else 0.0
            results.append(DecodedSequence(ids=collapsed, confidence=conf, raw_score=raw))
        return results
