"""LipNet 베이스라인: 3D-CNN + Bi-GRU + CTC head.

입력  : (B, C, T, H, W) = (B, 3, 75, 96, 96)  (RGB, ImageNet 정규화)
출력  : (T', B, V)                              CTC (time, batch, vocab)

baseline CER 40% 초과 시 backbone을 3D-ResNet으로 교체할 수 있도록
spatiotemporal feature 추출부와 sequence 모델링부를 분리.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from backend.config import CHANNELS


@dataclass
class LipNetConfig:
    in_channels: int = CHANNELS            # 3 (RGB) — CLAUDE 기준값
    conv_channels: tuple[int, int, int] = (32, 64, 96)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.5
    vocab_size: int = 41                   # 자음19 + 모음21 + blank


class _STCNN(nn.Module):
    """Spatiotemporal 3D-CNN feature extractor (LipNet 구조 참고)."""

    def __init__(self, in_c: int, channels: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.block1 = nn.Sequential(
            nn.Conv3d(in_c, c1, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class LipNet(nn.Module):
    def __init__(self, config: LipNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LipNetConfig()
        self.cnn = _STCNN(self.cfg.in_channels, self.cfg.conv_channels, self.cfg.dropout)

        # CNN 출력 feature dim은 forward 시점에 지연 결정 (LazyLinear 사용)
        self.rnn = nn.GRU(
            input_size=1,             # 임시값; forward에서 lazy 모듈로 채움
            hidden_size=self.cfg.rnn_hidden,
            num_layers=self.cfg.rnn_layers,
            bidirectional=True,
            dropout=self.cfg.dropout if self.cfg.rnn_layers > 1 else 0.0,
            batch_first=False,
        )
        # 실제로는 LazyLinear로 처리 (CNN 출력 dim을 모를 수 있으므로)
        self.proj_in = nn.LazyLinear(self.cfg.rnn_hidden * 2)  # CNN feat → RNN input
        # 위 self.rnn은 동적으로 재구성됨 — 아래 _ensure_rnn 참조
        self._rnn_ready = False
        self.head = nn.Linear(self.cfg.rnn_hidden * 2, self.cfg.vocab_size)

    def _ensure_rnn(self, feat_dim: int) -> None:
        if self._rnn_ready:
            return
        # proj_in으로 일정 차원(rnn_hidden*2)에 맞춘 뒤 GRU 입력으로 사용
        self.rnn = nn.GRU(
            input_size=self.cfg.rnn_hidden * 2,
            hidden_size=self.cfg.rnn_hidden,
            num_layers=self.cfg.rnn_layers,
            bidirectional=True,
            dropout=self.cfg.dropout if self.cfg.rnn_layers > 1 else 0.0,
            batch_first=False,
        ).to(next(self.parameters()).device)
        self._rnn_ready = True
        _ = feat_dim  # noqa: F841 — 추후 디버깅용

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) → logits (T', B, V) for CTC."""
        feat = self.cnn(x)                         # (B, C', T, H', W')
        b, c, t, h, w = feat.shape
        feat = feat.permute(2, 0, 1, 3, 4)         # (T, B, C', H', W')
        feat = feat.reshape(t, b, c * h * w)       # (T, B, F)
        feat = self.proj_in(feat)                  # (T, B, rnn_in)
        self._ensure_rnn(feat.shape[-1])
        out, _ = self.rnn(feat)                    # (T, B, 2*rnn_hidden)
        logits = self.head(out)                    # (T, B, V)
        return logits

    @staticmethod
    def compute_input_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        """CNN/Pool stride에 따른 시간축 다운샘플 비율을 반영.

        현재 _STCNN은 시간축 stride=1만 사용하므로 길이는 그대로.
        backbone을 바꾸면 이 함수를 수정한다.
        """
        return input_lengths.clone()
