"""MediaPipe Face Mesh 기반 입술 ROI 추출 (96×96 RGB).

backend/config.py 의 ROI_SIZE, ROI_MARGIN, ROI_CROP_IDX 를 단일 진실 공급원으로
참조한다. 임의 하드코딩 금지.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from backend.config import ROI_CROP_IDX, ROI_MARGIN, ROI_SIZE


@dataclass
class ExtractorConfig:
    out_size: tuple[int, int] = ROI_SIZE      # (H, W)
    padding_ratio: float = ROI_MARGIN
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    landmark_indices: list[int] = field(default_factory=lambda: list(ROI_CROP_IDX))


class LipROIExtractor:
    """프레임 → 입술 ROI RGB 이미지 (H, W, 3) uint8."""

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
        """프레임 → ROI RGB (H, W, 3). 얼굴 미검출 시 None."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        pts = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in self.cfg.landmark_indices],
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

        roi = rgb[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        roi = cv2.resize(roi, self.cfg.out_size, interpolation=cv2.INTER_AREA)
        return roi  # (H, W, 3) RGB uint8

    def extract_video(self, video_path: str) -> np.ndarray:
        """비디오 → (T, H, W, 3) RGB. 검출 실패 프레임은 last-frame 복제."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frames: list[np.ndarray] = []
        h, w = self.cfg.out_size
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                roi = self.extract(frame)
                if roi is None:
                    roi = frames[-1].copy() if frames else np.zeros((h, w, 3), dtype=np.uint8)
                frames.append(roi)
        finally:
            cap.release()

        return np.stack(frames, axis=0) if frames else np.zeros((0, h, w, 3), dtype=np.uint8)
