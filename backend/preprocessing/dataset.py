"""학습용 PyTorch Dataset (RGB + ImageNet normalize + last-frame 패딩)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.config import NORMALIZE_MEAN, NORMALIZE_STD, SEQ_LEN
from backend.models import GRAPHEME_VOCAB


@dataclass
class Sample:
    clip_path: Path
    text: str
    target_ids: list[int]


def _reassemble_syllables(jamos: list[str], j2h) -> str:
    """[ㅎ,ㅏ,ㄴ,ㄱ,ㅡ,ㄱ] → '한글'. 초성-중성-(종성?) 그리디 그룹핑."""
    consonants = set("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    vowels = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    out: list[str] = []
    i, n = 0, len(jamos)
    while i < n:
        if jamos[i] not in consonants:
            out.append(jamos[i])
            i += 1
            continue
        cho = jamos[i]
        if i + 1 >= n or jamos[i + 1] not in vowels:
            out.append(cho)
            i += 1
            continue
        jung = jamos[i + 1]
        jong: str | None = None
        if i + 2 < n and jamos[i + 2] in consonants:
            if i + 3 >= n or jamos[i + 3] not in vowels:
                jong = jamos[i + 2]
        try:
            out.append(j2h(cho, jung, jong) if jong else j2h(cho, jung))
        except (ValueError, TypeError):
            out.append(cho + jung + (jong or ""))
        i += 3 if jong else 2
    return "".join(out)


class JamoTokenizer:
    """한국어 자모 41클래스 (자음19 + 모음21 + blank) 토크나이저.

    - blank id = 0 (CTC blank)
    - encode: 한글 음절을 자모로 분해 후 인덱스
    - decode: 자모 인덱스 → 음절 재조립 (python-jamo j2h)
    """

    BLANK_ID = 0

    def __init__(self) -> None:
        self.id_to_jamo: list[str] = ["<blank>"] + list(GRAPHEME_VOCAB)
        self.jamo_to_id: dict[str, int] = {j: i for i, j in enumerate(self.id_to_jamo)}

    def encode(self, text: str) -> list[int]:
        try:
            from jamo import h2j, j2hcj
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("python-jamo 미설치: pip install jamo") from e
        decomposed = j2hcj(h2j(text))
        return [self.jamo_to_id[j] for j in decomposed if j in self.jamo_to_id]

    def decode(self, ids: list[int]) -> str:
        try:
            from jamo import j2h
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("python-jamo 미설치: pip install jamo") from e
        jamos = [
            self.id_to_jamo[i]
            for i in ids
            if i != self.BLANK_ID and 0 < i < len(self.id_to_jamo)
        ]
        return _reassemble_syllables(jamos, j2h)

    def __len__(self) -> int:
        return len(self.id_to_jamo)


class CharTokenizer(JamoTokenizer):
    """후방 호환 별칭. 신규 코드는 JamoTokenizer 사용 권장."""

    def __init__(self, vocab_path: str | Path | None = None) -> None:  # noqa: ARG002
        super().__init__()


class LipReadingDataset(Dataset):
    """JSONL manifest 기반 학습 샘플 로더.

    manifest 예: {"clip": "...npy", "text": "안녕하세요"}
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: JamoTokenizer,
        max_frames: int = SEQ_LEN,
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
        clip = np.load(s.clip_path)              # (T, H, W, 3) RGB uint8
        if clip.ndim == 3:                       # grayscale → 3채널로 broadcast
            clip = np.repeat(clip[..., None], 3, axis=-1)
        clip = clip[: self.max_frames]

        # ImageNet normalize
        clip = clip.astype(np.float32) / 255.0
        mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
        std = np.array(NORMALIZE_STD, dtype=np.float32)
        clip = (clip - mean) / std               # (T, H, W, 3)

        # (T, H, W, C) → (C, T, H, W)
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
        target = torch.tensor(s.target_ids, dtype=torch.long)
        return tensor, target


def collate_ctc(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 길이 (C,T,H,W) 시퀀스를 last-frame 패딩으로 묶는다 (zero-padding 금지)."""
    clips, targets = zip(*batch, strict=True)
    c, _, h, w = clips[0].shape
    input_lengths = torch.tensor([clip.shape[1] for clip in clips], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    t_max = int(input_lengths.max().item())
    padded = torch.empty(len(clips), c, t_max, h, w, dtype=torch.float32)
    for i, clip in enumerate(clips):
        t = clip.shape[1]
        padded[i, :, :t] = clip
        if t < t_max:
            last = clip[:, -1:, :, :].expand(c, t_max - t, h, w)
            padded[i, :, t:] = last

    targets_concat = torch.cat(targets) if len(targets) > 0 else torch.empty(0, dtype=torch.long)
    return padded, targets_concat, input_lengths, target_lengths
