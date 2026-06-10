"""테스트셋 CER/WER 평가.

baseline CER가 model.yaml의 baseline_cer_threshold(기본 0.40)를 넘으면
3D-ResNet backbone으로 교체할 시점이라는 신호로 사용한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from backend.models import GreedyCTCDecoder, LipNet, LipNetConfig
from backend.preprocessing import LipReadingDataset, collate_ctc
from backend.preprocessing.dataset import CharTokenizer


def cer(reference: str, hypothesis: str) -> float:
    """문자 단위 Levenshtein 거리 / 참조 문자수."""
    r, h = list(reference), list(hypothesis)
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(r)][len(h)] / max(1, len(r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.data_config, encoding="utf-8") as f:
        cfg_data = yaml.safe_load(f)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    cfg_model = ckpt["config"]

    tokenizer = CharTokenizer(cfg_data["dataset"]["vocab_path"])
    test_ds = LipReadingDataset(
        manifest_path=cfg_data["dataset"]["manifest_test"],
        tokenizer=tokenizer,
        max_frames=cfg_data["preprocess"]["max_frames"],
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=cfg_data["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=cfg_data["dataloader"]["num_workers"],
        collate_fn=collate_ctc,
    )

    model_cfg = LipNetConfig(
        in_channels=cfg_model["model"]["in_channels"],
        conv_channels=tuple(cfg_model["model"]["conv_channels"]),
        rnn_hidden=cfg_model["model"]["rnn_hidden"],
        rnn_layers=cfg_model["model"]["rnn_layers"],
        dropout=0.0,
        vocab_size=len(tokenizer),
    )
    model = LipNet(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    decoder = GreedyCTCDecoder(blank_id=CharTokenizer.BLANK_ID)
    total_cer, n = 0.0, 0
    with torch.no_grad():
        for clips, targets, in_lens, tgt_lens in test_dl:
            clips = clips.to(args.device)
            logits = model(clips)
            preds = decoder(logits)
            # 정답 분해 (concat → split)
            offset = 0
            for i, tlen in enumerate(tgt_lens.tolist()):
                ref_ids = targets[offset : offset + tlen].tolist()
                offset += tlen
                ref = tokenizer.decode(ref_ids)
                hyp = tokenizer.decode(preds[i])
                total_cer += cer(ref, hyp)
                n += 1

    avg = total_cer / max(1, n)
    threshold = cfg_model.get("eval", {}).get("baseline_cer_threshold", 0.40)
    print(f"Test CER: {avg:.4f}  (n={n})")
    if avg > threshold:
        print(f"WARNING: CER {avg:.4f} > threshold {threshold:.2f} -> switch backbone to 3D-ResNet")


if __name__ == "__main__":
    main()
