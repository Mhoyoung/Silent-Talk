// 입술 ROI 크롭(96×96 RGB) + mouth_ratio 계산 헬퍼.
// CLAUDE.md 의 ROI/V-VAD 규칙을 그대로 반영한다.

import {
  LATERAL_LEFT_IDX,
  LATERAL_RIGHT_IDX,
  ROI_CROP_IDX,
  ROI_MARGIN,
  ROI_SIZE,
  VVAD_IDX_BOT,
  VVAD_IDX_TOP,
} from './constants.js';

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const norm = (v) => Math.hypot(v.x, v.y);

function centroid(landmarks, idxs) {
  let sx = 0, sy = 0;
  for (const i of idxs) { sx += landmarks[i].x; sy += landmarks[i].y; }
  return { x: sx / idxs.length, y: sy / idxs.length };
}

export function mouthRatio(landmarks) {
  const upper = centroid(landmarks, VVAD_IDX_TOP);
  const lower = centroid(landmarks, VVAD_IDX_BOT);
  const left  = landmarks[LATERAL_LEFT_IDX];
  const right = landmarks[LATERAL_RIGHT_IDX];
  const dRaw  = norm(sub(upper, lower));
  const width = norm(sub(left, right)) + 1e-6;
  return dRaw / width;
}

export function cropLipROI(video, landmarks, ctx) {
  const w = video.videoWidth, h = video.videoHeight;
  const xs = [], ys = [];
  for (const i of ROI_CROP_IDX) {
    xs.push(landmarks[i].x * w);
    ys.push(landmarks[i].y * h);
  }
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const bw = xMax - xMin, bh = yMax - yMin;
  const side = Math.max(bw * (1 + 2 * ROI_MARGIN), bh * (1 + 2 * ROI_MARGIN));
  const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;
  const x0 = Math.max(0, Math.min(w - side, cx - side / 2));
  const y0 = Math.max(0, Math.min(h - side, cy - side / 2));
  // RGB 그대로 (grayscale 변환 없음) — 서버는 PIL "RGB" 로 디코딩한다.
  ctx.drawImage(video, x0, y0, side, side, 0, 0, ROI_SIZE, ROI_SIZE);
}
