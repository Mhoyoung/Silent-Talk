"""PyTorch 체크포인트 → ONNX 변환 스크립트.

배포 시 ONNX Runtime로 추론 속도/이식성 확보.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from backend.models import LipNet, LipNetConfig
from backend.preprocessing.dataset import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("configs/vocab_ko_mvp.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--size", type=int, default=112)
    args = parser.parse_args()

    tokenizer = CharTokenizer(args.vocab)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    mc = ckpt["config"]["model"]
    cfg = LipNetConfig(
        in_channels=mc["in_channels"],
        conv_channels=tuple(mc["conv_channels"]),
        rnn_hidden=mc["rnn_hidden"],
        rnn_layers=mc["rnn_layers"],
        dropout=0.0,
        vocab_size=len(tokenizer),
    )
    model = LipNet(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, cfg.in_channels, args.frames, args.size, args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["clip"],
        output_names=["logits"],
        dynamic_axes={"clip": {0: "batch", 2: "time"}, "logits": {0: "time", 1: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
