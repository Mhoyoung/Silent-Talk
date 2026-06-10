"""Silent Talk 학습 엔트리포인트.

사용:
    python scripts/train.py --data-config configs/data.yaml --model-config configs/model.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.models import LipNet, LipNetConfig
from backend.preprocessing import LipReadingDataset, collate_ctc
from backend.preprocessing.dataset import CharTokenizer


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg_data: dict, tokenizer: CharTokenizer, split: str) -> DataLoader:
    manifest_key = f"manifest_{split}"
    ds = LipReadingDataset(
        manifest_path=cfg_data["dataset"][manifest_key],
        tokenizer=tokenizer,
        max_frames=cfg_data["preprocess"]["max_frames"],
    )
    dl_cfg = cfg_data["dataloader"]
    return DataLoader(
        ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=(split == "train") and dl_cfg["shuffle"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=collate_ctc,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg_data = load_yaml(args.data_config)
    cfg_model = load_yaml(args.model_config)

    tokenizer = CharTokenizer()
    train_dl = build_dataloader(cfg_data, tokenizer, "train")
    val_dl = build_dataloader(cfg_data, tokenizer, "val")

    model_cfg = LipNetConfig(
        in_channels=cfg_model["model"]["in_channels"],
        conv_channels=tuple(cfg_model["model"]["conv_channels"]),
        rnn_hidden=cfg_model["model"]["rnn_hidden"],
        rnn_layers=cfg_model["model"]["rnn_layers"],
        dropout=cfg_model["model"]["dropout"],
        vocab_size=len(tokenizer),
    )
    model = LipNet(model_cfg).to(args.device)

    tcfg = cfg_model["train"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )
    ctc_loss = nn.CTCLoss(blank=CharTokenizer.BLANK_ID, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=tcfg["amp"] and args.device == "cuda")

    ckpt_dir = Path(tcfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running = 0.0
        for step, (clips, targets, in_lens, tgt_lens) in enumerate(
            tqdm(train_dl, desc=f"epoch {epoch}")
        ):
            clips = clips.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)
            in_lens = LipNet.compute_input_lengths(in_lens).to(args.device)
            tgt_lens = tgt_lens.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=tcfg["amp"] and args.device == "cuda"):
                logits = model(clips)                              # (T, B, V)
                log_probs = logits.log_softmax(dim=-1)
                loss = ctc_loss(log_probs, targets, in_lens, tgt_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        avg_train = running / max(1, len(train_dl))

        # validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for clips, targets, in_lens, tgt_lens in val_dl:
                clips = clips.to(args.device)
                targets = targets.to(args.device)
                in_lens = LipNet.compute_input_lengths(in_lens).to(args.device)
                tgt_lens = tgt_lens.to(args.device)
                logits = model(clips)
                log_probs = logits.log_softmax(dim=-1)
                v_loss += ctc_loss(log_probs, targets, in_lens, tgt_lens).item()
        avg_val = v_loss / max(1, len(val_dl))

        print(f"[epoch {epoch}] train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({"model": model.state_dict(), "config": cfg_model}, ckpt_dir / "best.pt")
            print(f"  -> saved best to {ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    main()
