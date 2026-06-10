"""실시간 립리딩 WebSocket.

클라이언트 메시지 (JSON):
    {"type": "frame", "roi": <base64 PNG>, "mar": <float>}
    {"type": "reset"}

서버 메시지:
    {"type": "partial", "text": "..."}     # 발화 종료 시 1회
    {"type": "info", "msg": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import io

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from backend.config import settings
from backend.routers.inference import get_service
from backend.services.stream_buffer import StreamBuffer

router = APIRouter()


def _decode_roi(b64_png: str) -> np.ndarray:
    raw = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
    return np.array(img, dtype=np.uint8)


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    buffer = StreamBuffer(max_frames=settings.max_clip_frames)
    service = get_service()
    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "reset":
                buffer.reset()
                await ws.send_json({"type": "info", "msg": "buffer reset"})
                continue
            if msg.get("type") != "frame":
                continue

            try:
                roi = _decode_roi(msg["roi"])
                mar = float(msg.get("mar", 0.0))
            except (KeyError, ValueError):
                await ws.send_json({"type": "info", "msg": "invalid frame"})
                continue

            clip = buffer.push(roi, mar)
            if clip is None or clip.shape[0] < 8:
                continue

            # 추론은 블로킹 → 스레드풀로
            text = await loop.run_in_executor(None, service.infer, clip)
            await ws.send_json({"type": "partial", "text": text})

    except WebSocketDisconnect:
        return
