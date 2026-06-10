"""전역 설정 + 불변 기준값.

전체 파이프라인의 계약(contract). 임의 변경 시 전처리·모델·디코더·프론트엔드
미러 상수와 동시에 수정해야 한다.

런타임 가변 값(체크포인트 경로, device 등)은 ST_* 환경변수로 오버라이드
가능한 Settings 객체로 노출한다.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 절대 변경 금지 기준값
# ---------------------------------------------------------------------------

FPS: int = 25
"""목표 프레임 레이트. 비디오 캡처 fps 조작 금지 — 타임스탬프 기반 리샘플링 필수."""

SEQ_LEN: int = 75
"""시퀀스 길이 (3초 × 25fps)."""

ROI_SIZE: tuple[int, int] = (96, 96)
"""입술 ROI 크롭 출력 (H, W)."""

ROI_MARGIN: float = 0.20
"""ROI 바운딩 박스 상하좌우 margin 비율. margin 추가 후 정사각 resize."""

PADDING: str = "last-frame"
"""부족한 프레임 패딩 전략. zero-padding 금지 — 마지막 프레임 복제."""

PREPROC_SHAPE: tuple[str, ...] = ("B", "T", "H", "W", "C")
"""전처리 출력 텐서 형상 = (B, 75, 96, 96, 3)."""

MODEL_INPUT_SHAPE: tuple[str, ...] = ("B", "C", "T", "H", "W")
"""모델 입력 형상 = (B, 3, 75, 96, 96). permute((0, 4, 1, 2, 3)) 변환 필수."""

NORMALIZE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""ImageNet 정규화 평균 (R, G, B)."""

NORMALIZE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet 정규화 표준편차 (R, G, B)."""

CHANNELS: int = 3
"""입력 채널 수 (RGB)."""

# ---------------------------------------------------------------------------
# 랜드마크 인덱스 (MediaPipe Face Mesh)
# ---------------------------------------------------------------------------

ROI_CROP_IDX: list[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
"""ROI 크롭 기준 외측 입술 랜드마크 20개."""

VVAD_IDX_TOP: list[int] = [13, 14, 312, 311, 310, 415]
"""윗입술 내측 랜드마크 — d_raw 상단 중심점 계산."""

VVAD_IDX_BOT: list[int] = [17, 18, 84, 181, 180, 314]
"""아랫입술 내측 랜드마크 — d_raw 하단 중심점 계산.

d_raw = 윗입술 평균좌표 ↔ 아랫입술 평균좌표 유클리드 거리.
mouth_ratio = d_raw / W (W: 좌우 광대 랜드마크 거리).
"""

# ---------------------------------------------------------------------------
# 업로드 / 추론 제약
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE_MB: int = 500
"""업로드 허용 최대 파일 크기 (MB). 초과 시 400."""

MAX_DURATION_SEC: int = 180
"""업로드 영상 허용 최대 길이 (초)."""

JOB_TIMEOUT_SEC: int = 300
"""비동기 Job 타임아웃 (초)."""

POLL_INTERVAL_SEC: float = 1.0
"""GET /api/result 폴링 권장 간격 (초)."""

TEST_SETS: dict[str, str] = {
    # "demo_word": r"D:\009.립리딩(입모양) 음성인식 데이터\01.데이터\2.Validation",
}
"""평가용 테스트셋 화이트리스트 (test_set_id → 경로). 경로 직접 전달 금지."""

# ---------------------------------------------------------------------------
# CORS / 경로
# ---------------------------------------------------------------------------

CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
"""CORS 허용 오리진 화이트리스트. 와일드카드 '*' 금지."""

TMP_UPLOAD_DIR: str = os.path.join(os.getcwd(), "tmp", "uploads")
"""임시 업로드 파일 저장 디렉토리. 추론 완료 후 즉시 os.remove()."""

MODEL_WEIGHTS_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "weights", "baseline.pt"
)
"""베이스라인 모델 가중치 경로. 없으면 미학습 폴백.

*.pt는 .gitignore 대상 — 별도 배포."""

LOG_DIR: str = os.path.join(os.getcwd(), "logs")
"""로그 출력 디렉토리."""


# ---------------------------------------------------------------------------
# 런타임 설정 (env 오버라이드)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ST_", env_file=".env", extra="ignore")

    checkpoint: Path = Path(MODEL_WEIGHTS_PATH)
    onnx_path: Path | None = None
    device: str = "cpu"
    max_clip_frames: int = SEQ_LEN
    cors_origins: list[str] = CORS_ORIGINS
    tmp_upload_dir: str = TMP_UPLOAD_DIR


settings = Settings()
