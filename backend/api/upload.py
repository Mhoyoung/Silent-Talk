"""업로드 라우터 (stub).

설계상 POST /api/upload 는 영상 파일(<=500MB, <=180s)을 받아
MIME 2중 검증 후 임시 디렉터리에 저장하고 session_id를 발급한다.
실제 구현은 후속 단계에서 추가한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile) -> dict:
    raise HTTPException(status_code=501, detail="upload not implemented yet")
