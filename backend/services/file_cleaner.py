"""임시 업로드 파일 정리 스케줄러 (stub).

APScheduler가 설치돼 있으면 백그라운드에서 주기 정리를 수행하고,
없으면 no-op 으로 폴백한다 (개발 환경 호환).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_scheduler = None


def _sweep(directory: str | os.PathLike[str], older_than_sec: int = 3600) -> int:
    p = Path(directory)
    if not p.exists():
        return 0
    cutoff = time.time() - older_than_sec
    removed = 0
    for child in p.iterdir():
        try:
            if child.is_file() and child.stat().st_mtime < cutoff:
                child.unlink()
                removed += 1
        except OSError as e:
            logger.warning("failed to remove %s: %s", child, e)
    return removed


def start_cleanup_scheduler() -> None:
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
    except ImportError:
        logger.info("apscheduler not installed; file cleanup disabled")
        return

    from backend.config import settings

    target_dir = getattr(settings, "tmp_upload_dir", os.path.join(os.getcwd(), "tmp", "uploads"))
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(_sweep, "interval", minutes=10, args=[target_dir])
    scheduler.start()
    _scheduler = scheduler


def shutdown_cleanup_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
