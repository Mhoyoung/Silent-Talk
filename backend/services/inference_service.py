"""모델 로드 + 추론 서비스.

TorchScript / ONNX Runtime / 일반 PyTorch 체크포인트를 모두 지원.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch

from backend.config import NORMALIZE_MEAN, NORMALIZE_STD, settings
from backend.models import VOCAB_SIZE, GreedyCTCDecoder, LipNet, LipNetConfig
from backend.preprocessing.dataset import JamoTokenizer


class InferenceService:
    def __init__(self) -> None:
        self._lock = Lock()
        self.tokenizer = JamoTokenizer()
        self.decoder = GreedyCTCDecoder(blank_id=JamoTokenizer.BLANK_ID)
        self.device = torch.device(settings.device)
        self._onnx_session = None
        self._torch_model: LipNet | None = None
        self._load()

    def _load(self) -> None:
        if settings.onnx_path and Path(settings.onnx_path).exists():
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if settings.device == "cuda" else ["CPUExecutionProvider"]
            self._onnx_session = ort.InferenceSession(str(settings.onnx_path), providers=providers)
            return

        if not Path(settings.checkpoint).exists():
            # 가중치가 없으면 미학습 폴백 — 파이프라인 구조는 동작, 출력은 무의미
            self._torch_model = LipNet(LipNetConfig(vocab_size=VOCAB_SIZE)).to(self.device)
            self._torch_model.eval()
            return

        ckpt = torch.load(settings.checkpoint, map_location=self.device)
        mc = ckpt["config"]["model"]
        cfg = LipNetConfig(
            in_channels=mc["in_channels"],
            conv_channels=tuple(mc["conv_channels"]),
            rnn_hidden=mc["rnn_hidden"],
            rnn_layers=mc["rnn_layers"],
            dropout=0.0,
            vocab_size=len(self.tokenizer),
        )
        model = LipNet(cfg).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._torch_model = model

    def infer(self, clip: np.ndarray) -> str:
        """clip: (T, H, W, 3) RGB uint8 → 한국어 텍스트.

        grayscale (T,H,W) 입력은 3채널로 broadcast 후 처리한다 (호환성).
        """
        if clip.ndim == 3:
            clip = np.repeat(clip[..., None], 3, axis=-1)
        # ImageNet normalize
        clip = clip.astype(np.float32) / 255.0
        mean = np.array(NORMALIZE_MEAN, dtype=np.float32)
        std = np.array(NORMALIZE_STD, dtype=np.float32)
        clip = (clip - mean) / std
        # (T,H,W,C) → (B,C,T,H,W)
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        with self._lock:
            if self._onnx_session is not None:
                ort_inputs = {self._onnx_session.get_inputs()[0].name: tensor.numpy()}
                logits = torch.from_numpy(self._onnx_session.run(None, ort_inputs)[0])
            else:
                assert self._torch_model is not None
                with torch.no_grad():
                    logits = self._torch_model(tensor.to(self.device)).cpu()

        ids = self.decoder(logits)[0]
        return self.tokenizer.decode(ids)
