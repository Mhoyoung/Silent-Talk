"""실시간 립리딩 WebSocket 핸들러.

main.py가 등록한 /ws/stream 엔드포인트의 실제 처리를 담당한다.

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
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

from backend.api.infer import get_service
from backend.config import settings
from backend.services.stream_buffer import StreamBuffer


def _decode_roi(b64_png: str) -> np.ndarray:
    raw = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
    return np.array(img, dtype=np.uint8)


async def handle_connection(ws: WebSocket) -> None:
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

            text = await loop.run_in_executor(None, service.infer, clip)
            await ws.send_json({"type": "partial", "text": text})

    except WebSocketDisconnect:
        return
