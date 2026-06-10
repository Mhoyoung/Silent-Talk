"""Visual VAD + 시퀀스 세그먼터.

발화 구간 분리 규칙:
    d_raw       = 윗입술 평균좌표 ↔ 아랫입술 평균좌표 유클리드 거리
    mouth_ratio = d_raw / W    (W: 좌우 광대 랜드마크 거리)
    delta(t)    = |mouth_ratio[t] − 이동평균|
    active 구간 = delta(t) > threshold
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import SEQ_LEN, VVAD_IDX_BOT, VVAD_IDX_TOP


def mouth_ratio(landmarks_xy: np.ndarray, lateral_lr_xy: np.ndarray) -> float:
    """단일 프레임의 mouth_ratio = d_raw / W."""
    upper = landmarks_xy[VVAD_IDX_TOP].mean(axis=0)
    lower = landmarks_xy[VVAD_IDX_BOT].mean(axis=0)
    d_raw = float(np.linalg.norm(upper - lower))
    width = float(np.linalg.norm(lateral_lr_xy[0] - lateral_lr_xy[1])) + 1e-6
    return d_raw / width


@dataclass
class VVADConfig:
    window: int = 5
    mar_threshold: float = 0.04
    min_speech_frames: int = 6
    min_silence_frames: int = 4


class VisualVAD:
    """mouth_ratio 시계열 → 발화 구간 [start, end) 리스트."""

    def __init__(self, config: VVADConfig | None = None) -> None:
        self.cfg = config or VVADConfig()

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        w = self.cfg.window
        if w <= 1 or values.size < w:
            return values
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(values, kernel, mode="same")

    def segments(self, mr_sequence: np.ndarray) -> list[tuple[int, int]]:
        if mr_sequence.size == 0:
            return []

        smoothed = self._smooth(mr_sequence.astype(np.float32))
        delta = np.abs(np.diff(smoothed, prepend=smoothed[0]))
        active = delta > self.cfg.mar_threshold

        segments: list[tuple[int, int]] = []
        i = 0
        n = active.size
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            silence_run = 0
            while j < n:
                if active[j]:
                    silence_run = 0
                    j += 1
                else:
                    silence_run += 1
                    if silence_run >= self.cfg.min_silence_frames:
                        break
                    j += 1
            end = j - silence_run
            if end - i >= self.cfg.min_speech_frames:
                segments.append((i, end))
            i = j
        return segments


def split_into_seq_chunks(clip: np.ndarray, seq_len: int = SEQ_LEN) -> list[np.ndarray]:
    """긴 클립을 seq_len 단위로 분할. 부족한 마지막 청크는 last-frame 패딩 (zero 금지)."""
    if clip.shape[0] == 0:
        return []

    chunks: list[np.ndarray] = []
    for start in range(0, clip.shape[0], seq_len):
        chunk = clip[start : start + seq_len]
        if chunk.shape[0] < seq_len:
            pad_len = seq_len - chunk.shape[0]
            last = chunk[-1:].repeat(pad_len, axis=0)
            chunk = np.concatenate([chunk, last], axis=0)
        chunks.append(chunk)
    return chunks
