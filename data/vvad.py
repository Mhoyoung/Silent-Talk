"""Visual Voice Activity Detection (V-VAD).

입술 ROI 시퀀스에서 발화 구간을 자동으로 분리한다.
"입이 얼마나 벌어졌는가"(MAR; Mouth Aspect Ratio) 변화량을
이동평균으로 평활화해 임계값 기반으로 발화/무음을 결정한다.

오디오 기반 VAD가 불가능한 무음 환경(립리딩)을 위한 핵심 모듈.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Face Mesh 기준 윗입술(중앙), 아랫입술(중앙), 좌/우 입꼬리 인덱스
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER = 78
RIGHT_MOUTH_CORNER = 308


@dataclass
class VVADConfig:
    window: int = 5                # 이동평균 윈도우 (frames)
    mar_threshold: float = 0.04    # 발화로 간주할 MAR 변화량 임계값
    min_speech_frames: int = 6     # 최소 발화 길이 (frames)
    min_silence_frames: int = 4    # 발화 사이 최소 무음 길이


def mouth_aspect_ratio(landmarks_xy: np.ndarray) -> float:
    """MAR = (입 세로 거리) / (입 가로 거리). 단일 프레임 기준."""
    upper = landmarks_xy[UPPER_LIP_CENTER]
    lower = landmarks_xy[LOWER_LIP_CENTER]
    left = landmarks_xy[LEFT_MOUTH_CORNER]
    right = landmarks_xy[RIGHT_MOUTH_CORNER]
    vertical = float(np.linalg.norm(upper - lower))
    horizontal = float(np.linalg.norm(left - right)) + 1e-6
    return vertical / horizontal


class VisualVAD:
    """프레임별 MAR 시퀀스를 입력으로 받아 발화 구간 [start, end)을 반환."""

    def __init__(self, config: VVADConfig | None = None) -> None:
        self.cfg = config or VVADConfig()

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        w = self.cfg.window
        if w <= 1 or values.size < w:
            return values
        kernel = np.ones(w, dtype=np.float32) / w
        return np.convolve(values, kernel, mode="same")

    def segments(self, mar_sequence: np.ndarray) -> list[tuple[int, int]]:
        """MAR 시계열 → [(start, end), ...] 발화 구간 인덱스."""
        if mar_sequence.size == 0:
            return []

        smoothed = self._smooth(mar_sequence.astype(np.float32))
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
