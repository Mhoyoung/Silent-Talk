"""PyTorch Dataset for lip reading.

전처리된 입술 ROI 시퀀스(npy)와 라벨(텍스트)을 묶어 학습용 배치를 제공한다.
CTC loss를 사용하므로 collate에서 (input_lengths, target_lengths)도 함께 반환한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    clip_path: Path          # 전처리된 (T, H, W) 또는 (T, H, W, 3) npy
    text: str                # 원본 한국어 텍스트
    target_ids: list[int]    # vocab으로 인코딩된 토큰 시퀀스


class CharTokenizer:
    """한국어 음절 단위(또는 어절 단위) 토크나이저.

    1차 MVP에서는 vocab 파일(JSON: char -> id)에 등장한 문자만 다루며,
    OOV는 무시한다. CTC blank는 항상 id=0으로 둔다.
    """

    BLANK_ID = 0

    def __init__(self, vocab_path: str | Path) -> None:
        with open(vocab_path, encoding="utf-8") as f:
            self.char_to_id: dict[str, int] = json.load(f)
        if self.char_to_id.get("<blank>", 0) != 0:
            raise ValueError("Vocab must reserve id=0 for <blank>.")
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids if i != self.BLANK_ID)

    def __len__(self) -> int:
        return len(self.char_to_id)


class LipReadingDataset(Dataset):
    """manifest(JSONL)로 학습 샘플을 로드.

    manifest 예:
        {"clip": "data/processed/spk01/0001.npy", "text": "안녕하세요"}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharTokenizer,
        max_frames: int = 75,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.samples: list[Sample] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.samples.append(
                    Sample(
                        clip_path=Path(rec["clip"]),
                        text=rec["text"],
                        target_ids=tokenizer.encode(rec["text"]),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        clip = np.load(s.clip_path)              # (T, H, W) or (T, H, W, 3)
        if clip.ndim == 3:
            clip = clip[..., None]               # (T, H, W, 1)
        clip = clip[: self.max_frames]
        clip = clip.astype(np.float32) / 255.0
        # to (C, T, H, W) — 3D CNN 입력 포맷
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
        target = torch.tensor(s.target_ids, dtype=torch.long)
        return tensor, target


def collate_ctc(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 길이 (C,T,H,W) 시퀀스를 시간축에 0-padding 한 배치로 묶는다."""
    clips, targets = zip(*batch, strict=True)
    c, _, h, w = clips[0].shape
    input_lengths = torch.tensor([clip.shape[1] for clip in clips], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    t_max = int(input_lengths.max().item())
    padded = torch.zeros(len(clips), c, t_max, h, w, dtype=torch.float32)
    for i, clip in enumerate(clips):
        padded[i, :, : clip.shape[1]] = clip

    targets_concat = torch.cat(targets) if len(targets) > 0 else torch.empty(0, dtype=torch.long)
    return padded, targets_concat, input_lengths, target_lengths
