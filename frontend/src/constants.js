// CLAUDE.md 불변 기준값 미러 — backend/config.py 와 동기 유지 필수.

export const FPS = 25;
export const SEQ_LEN = 75;                  // 3s × 25fps
export const ROI_SIZE = 96;                 // 96×96 정사각
export const ROI_MARGIN = 0.20;             // bbox 상하좌우 20% 여백
export const PADDING = 'last-frame';

export const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
export const NORMALIZE_STD  = [0.229, 0.224, 0.225];

// MediaPipe Face Mesh 입술 외측 20개
export const ROI_CROP_IDX = [
  61, 185,  40,  39,  37,   0, 267, 269, 270, 409,
 291, 375, 321, 405, 314,  17,  84, 181,  91, 146,
];

// V-VAD 윗/아랫입술 내측 — d_raw 중심점
export const VVAD_IDX_TOP = [13, 14, 312, 311, 310, 415];
export const VVAD_IDX_BOT = [17, 18,  84, 181, 180, 314];

// 좌우 광대 (mouth_ratio = d_raw / W 의 W)
export const LATERAL_LEFT_IDX  = 234;
export const LATERAL_RIGHT_IDX = 454;
