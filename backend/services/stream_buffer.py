"""실시간 프레임 스트림 → 발화 클립 버퍼.

WebSocket으로 들어온 입술 ROI 프레임을 누적하다가, V-VAD가 발화 종료를
감지하면 누적 클립을 추론 큐에 넘긴다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.preprocessing.segmenter import VVADConfig, VisualVAD


@dataclass
class StreamBuffer:
    max_frames: int = 75
    vvad: VisualVAD = field(default_factory=lambda: VisualVAD(VVADConfig()))
    _frames: deque[np.ndarray] = field(default_factory=deque)
    _mar_history: deque[float] = field(default_factory=deque)
    _silence_run: int = 0
    _in_speech: bool = False

    def push(self, frame: np.ndarray, mar: float) -> np.ndarray | None:
        """프레임 누적. 발화 종료가 감지되면 누적 클립(T,H,W)을 반환하고 비움."""
        self._frames.append(frame)
        self._mar_history.append(mar)
        if len(self._frames) > self.max_frames:
            self._frames.popleft()
            self._mar_history.popleft()

        if len(self._mar_history) < self.vvad.cfg.window + 1:
            return None

        recent = np.array(list(self._mar_history)[-(self.vvad.cfg.window + 1):], dtype=np.float32)
        delta = float(np.abs(np.diff(recent)).mean())
        active = delta > self.vvad.cfg.mar_threshold

        if active:
            self._in_speech = True
            self._silence_run = 0
            return None

        if self._in_speech:
            self._silence_run += 1
            if self._silence_run >= self.vvad.cfg.min_silence_frames:
                clip = np.stack(list(self._frames), axis=0)
                self.reset()
                return clip
        return None

    def reset(self) -> None:
        self._frames.clear()
        self._mar_history.clear()
        self._silence_run = 0
        self._in_speech = False
