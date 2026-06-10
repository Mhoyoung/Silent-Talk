"""API 요청/응답 Pydantic 스키마."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    size_bytes: int
    duration_sec: float


class InferRequest(BaseModel):
    session_id: str


class InferAccepted(BaseModel):
    job_id: str
    session_id: str
    status: Literal["processing"] = "processing"


class SegmentResult(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_score: float


class InferResult(BaseModel):
    session_id: str
    status: Literal["processing", "done", "error", "timeout"]
    segments: list[SegmentResult] = []
    detail: str | None = None
