"""비동기 Job 상태 저장소 (stub, in-memory).

후속 단계: 업로드 추론 작업의 (job_id, session_id, status, result) 추적.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal


JobStatus = Literal["processing", "done", "error", "timeout"]


@dataclass
class Job:
    job_id: str
    session_id: str
    status: JobStatus = "processing"
    result: Any | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, Job] = {}

    def create(self, job_id: str, session_id: str) -> Job:
        with self._lock:
            job = Job(job_id=job_id, session_id=session_id)
            self._jobs[session_id] = job
            return job

    def get(self, session_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(session_id)

    def update(self, session_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            return job


_manager = JobManager()


def get_job_manager() -> JobManager:
    return _manager
