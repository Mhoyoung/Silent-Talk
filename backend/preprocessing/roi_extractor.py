"""Lip ROI extraction using MediaPipe Face Mesh.

MediaPipe Face Mesh의 468개 랜드마크 중 입술 외곽 인덱스를 사용해
영상의 매 프레임마다 입술 영역을 잘라 고정 크기로 리사이즈한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 기준 입술 외곽선 랜드마크 인덱스
LIP_LANDMARKS: tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
)


@dataclass
class ExtractorConfig:
    out_size: int = 112             # 모델 입력 크기 (정사각)
    padding_ratio: float = 0.25     # ROI bbox 여백 비율
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    grayscale: bool = True          # LipNet 계열은 grayscale 입력 사용


class LipROIExtractor:
    """프레임 → 입술 ROI 이미지 (H, W) 또는 (H, W, 3)."""

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self.cfg = config or ExtractorConfig()
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "LipROIExtractor":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """프레임에서 입술 ROI를 잘라 반환. 얼굴 미검출 시 None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in LIP_LANDMARKS],
            dtype=np.float32,
        )
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        bw, bh = x_max - x_min, y_max - y_min
        pad_x = bw * self.cfg.padding_ratio
        pad_y = bh * self.cfg.padding_ratio

        side = max(bw + 2 * pad_x, bh + 2 * pad_y)
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        x0 = int(np.clip(cx - side / 2, 0, w - 1))
        y0 = int(np.clip(cy - side / 2, 0, h - 1))
        x1 = int(np.clip(cx + side / 2, 0, w))
        y1 = int(np.clip(cy + side / 2, 0, h))

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        roi = cv2.resize(roi, (self.cfg.out_size, self.cfg.out_size), interpolation=cv2.INTER_AREA)
        if self.cfg.grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi

    def extract_video(self, video_path: str) -> np.ndarray:
        """비디오 → (T, H, W) 또는 (T, H, W, 3). 검출 실패 프레임은 0으로 채움."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                roi = self.extract(frame)
                if roi is None:
                    shape = (self.cfg.out_size, self.cfg.out_size)
                    if not self.cfg.grayscale:
                        shape = (*shape, 3)
                    roi = np.zeros(shape, dtype=np.uint8)
                frames.append(roi)
        finally:
            cap.release()

        return np.stack(frames, axis=0)
