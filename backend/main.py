"""FastAPI 진입점.

CORS 화이트리스트, /api 라우터 등록, 라이프사이클(임시파일 정리 스케줄러),
/health 헬스체크, /ws/stream WebSocket을 구성한다. 실제 라우트 로직은
backend.api 패키지와 backend.services.ws_handler 에 위임한다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import get_api_router
from backend.config import settings
from backend.services.file_cleaner import (
    shutdown_cleanup_scheduler,
    start_cleanup_scheduler,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    start_cleanup_scheduler()
    try:
        yield
    finally:
        shutdown_cleanup_scheduler()


def create_app() -> FastAPI:
    app = FastAPI(title="Silent Talk Lip Reading API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(get_api_router())

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/stream")
    async def ws_stream(websocket: WebSocket) -> None:
        from backend.services.ws_handler import handle_connection
        await handle_connection(websocket)

    return app


app = create_app()
