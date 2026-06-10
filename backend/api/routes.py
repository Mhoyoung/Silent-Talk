"""/api prefix 아래로 모든 서브 라우터를 묶는 통합 라우터."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.evaluation import router as evaluation_router
from backend.api.infer import router as infer_router
from backend.api.upload import router as upload_router


def get_api_router() -> APIRouter:
    router = APIRouter(prefix="/api")
    router.include_router(upload_router, tags=["upload"])
    router.include_router(infer_router, tags=["infer"])
    router.include_router(evaluation_router, tags=["evaluation"])
    return router
