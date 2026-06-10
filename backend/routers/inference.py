"""단발 추론 REST 엔드포인트.

웹캠 영상 전체(혹은 ROI 시퀀스 npy)를 한 번에 업로드해 텍스트로 변환.
실시간 스트리밍은 routers/ws.py 참조.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["inference"])
_service: InferenceService | None = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


class InferResponse(BaseModel):
    text: str
    frames: int


@router.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(..., description="ROI npy (T,H,W) uint8")) -> InferResponse:
    raw = await file.read()
    try:
        clip = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid npy: {e}") from e
    if clip.ndim not in (3, 4):
        raise HTTPException(status_code=400, detail="Expected shape (T,H,W) or (T,H,W,C)")

    text = get_service().infer(clip)
    return InferResponse(text=text, frames=int(clip.shape[0]))
