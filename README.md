# Silent Talk

립리딩(Lip Reading) 기반 **실시간 텍스트 변환 시스템**.
웹캠으로 화자의 입술을 추적하고, 3D-CNN + LSTM + CTC 모델로 발화 내용을 한국어 텍스트로 변환합니다.

> 음성을 사용할 수 없는 환경(소음, 청각장애, 보안)에서 **소리 없이 의사소통**할 수 있도록 돕는 것을 목표로 합니다.

---

## 핵심 기능

- 실시간 웹캠 입술 ROI 추출 (MediaPipe Face Mesh)
- Visual VAD(V-VAD)로 발화 구간 자동 분할
- 3D-CNN + Bi-LSTM + CTC 기반 한국어 립리딩
- FastAPI + WebSocket을 통한 저지연 스트리밍 추론
- React + Tailwind 기반 실시간 자막 UI

---

## 1차 MVP 범위

| 구분 | 범위 |
| --- | --- |
| 학습 데이터 | AI Hub 한국어 립리딩 데이터셋 일부 (화자 N명) |
| 어휘 / 문장 | 핵심 어휘 100종 + 지정 문장 50개 내외 |
| 모델 (필수) | 3D-CNN + LSTM + CTC (LipNet 구조 참고) |
| 개선 기준 | baseline CER 40% 초과 시 3D-ResNet 우선 적용, 그 외에는 Transformer Encoder 실험 |
| 자유 발화 | 확장 목표 — MVP 범위 제외 |

---

## 기술 스택

**AI / 학습**
- Python 3.10+, PyTorch, OpenCV, MediaPipe
- 학습 환경: 로컬 PC / Google Colab Pro / 학교 GPU 서버

**백엔드**
- FastAPI, Uvicorn, WebSockets (비동기 프레임 스트리밍)

**프론트엔드**
- React.js (또는 Vanilla JS) + Tailwind CSS
- MediaPipe JS (클라이언트단 V-VAD / ROI 크롭)

**배포 / 최적화**
- Docker, ONNX Runtime, TorchScript

---

## 디렉터리 구조

```
Silent-Talk/
├── backend/
│   ├── main.py              FastAPI 진입점 (lifespan, /health, /ws/stream)
│   ├── config.py            전역 상수 (FPS/SEQ_LEN/ROI/CORS/제약)
│   ├── api/                 REST 라우트 (/api/* 통합)
│   │   ├── routes.py        ·  서브 라우터 집약 (upload/infer/evaluation)
│   │   ├── upload.py        ·  POST /api/upload (stub)
│   │   ├── infer.py         ·  POST /api/infer (현재 sync)
│   │   └── evaluation.py    ·  POST /api/evaluation/run (stub)
│   ├── schemas/models.py    Pydantic 요청/응답 스키마
│   ├── preprocessing/       전처리 파이프라인
│   │   ├── roi_extractor.py ·  MediaPipe 입술 ROI 크롭/리사이즈
│   │   ├── segmenter.py     ·  V-VAD 발화 구간 분리
│   │   └── dataset.py       ·  PyTorch Dataset + CTC collate
│   ├── models/              모델/디코더
│   │   ├── baseline.py      ·  LipNet (3D-CNN + BiGRU + CTC)
│   │   ├── decoder.py       ·  CTC 디코더
│   │   └── weights/         ·  *.pt (gitignored)
│   ├── services/
│   │   ├── inference_service.py
│   │   ├── ws_handler.py    ·  /ws/stream 실시간 핸들러
│   │   ├── stream_buffer.py ·  V-VAD 기반 발화 누적 버퍼
│   │   ├── job_manager.py   ·  비동기 Job 상태 (stub)
│   │   └── file_cleaner.py  ·  임시 업로드 정리 스케줄러
│   └── tests/
├── frontend/         React + Tailwind UI
├── configs/          학습 하이퍼파라미터 YAML
├── scripts/          extract_lip_rois / train / evaluate / export_onnx
├── docker/           Dockerfile.backend · Dockerfile.frontend · nginx.conf
└── docker-compose.yml
```

---

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 입술 ROI 추출
python scripts/extract_lip_rois.py --videos data_cache/videos --labels data_cache/labels.csv \
    --output data_cache/processed --manifest data_cache/manifest_train.jsonl

# 3. 학습
python scripts/train.py --config configs/model.yaml

# 4. 백엔드 실행
uvicorn backend.main:app --reload --port 8000

# 5. 프론트엔드 실행
cd frontend && npm install && npm run dev
```

Docker로 한 번에 실행하려면:

```bash
docker compose up --build
```

---

## 로드맵

- [x] 프로젝트 스캐폴딩
- [ ] 데이터 전처리 파이프라인 (MediaPipe ROI + V-VAD)
- [ ] 3D-CNN + LSTM + CTC 모델 학습 / 평가
- [ ] 실시간 추론 백엔드 (FastAPI + WebSocket)
- [ ] React 프론트엔드 + Docker 통합
- [ ] (확장) 자유 발화 인식, 화자 적응
