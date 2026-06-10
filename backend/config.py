"""런타임 설정 (env 기반)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ST_", env_file=".env", extra="ignore")

    checkpoint: Path = Path("checkpoints/lipnet_mvp/best.pt")
    onnx_path: Path | None = None         # 지정 시 ONNX Runtime 사용
    vocab_path: Path = Path("configs/vocab_ko_mvp.json")
    device: str = "cpu"                   # "cuda" 가능
    max_clip_frames: int = 75
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
