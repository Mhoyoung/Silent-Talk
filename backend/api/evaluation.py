"""평가 라우터 (stub).

POST /api/evaluation/run / GET /api/evaluation/status/{id} 를 노출할 예정.
test_set_id 화이트리스트 기반(경로 직접 전달 금지) — backend/config.py 의
TEST_SETS 에 등록된 식별자만 허용한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/evaluation/run")
async def run_evaluation(test_set_id: str, model_version: str) -> dict:
    raise HTTPException(status_code=501, detail="evaluation not implemented yet")


@router.get("/evaluation/status/{eval_id}")
async def evaluation_status(eval_id: str) -> dict:
    raise HTTPException(status_code=501, detail="evaluation not implemented yet")
