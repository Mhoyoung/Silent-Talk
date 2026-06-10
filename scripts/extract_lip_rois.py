"""원본 비디오 → 입술 ROI npy + manifest(JSONL) 생성.

ROI 크기/색상/패딩은 backend/config.py 의 ROI_SIZE/CHANNELS 를 따른다.

사용:
    python scripts/extract_lip_rois.py \
        --videos data_cache/videos \
        --labels data_cache/labels.csv \
        --output data_cache/processed \
        --manifest data_cache/manifest_train.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from backend.preprocessing.roi_extractor import ExtractorConfig, LipROIExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True, help="원본 비디오 디렉터리")
    parser.add_argument("--labels", type=Path, required=True, help="CSV: video_path,text")
    parser.add_argument("--output", type=Path, required=True, help="ROI npy 저장 디렉터리")
    parser.add_argument("--manifest", type=Path, required=True, help="manifest JSONL 출력 경로")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    cfg = ExtractorConfig()    # backend.config 의 ROI_SIZE/ROI_MARGIN 사용
    written = 0
    with open(args.labels, encoding="utf-8") as f, \
         open(args.manifest, "w", encoding="utf-8") as out, \
         LipROIExtractor(cfg) as extractor:
        reader = csv.DictReader(f)
        for row in tqdm(list(reader), desc="extract"):
            video_path = args.videos / row["video_path"]
            text = row["text"].strip()
            if not video_path.exists() or not text:
                continue

            clip = extractor.extract_video(str(video_path))
            if clip.shape[0] == 0:
                continue

            rel = video_path.relative_to(args.videos).with_suffix(".npy")
            dst = args.output / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, clip)

            out.write(json.dumps({"clip": str(dst), "text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. {written} samples written to {args.manifest}")


if __name__ == "__main__":
    main()
