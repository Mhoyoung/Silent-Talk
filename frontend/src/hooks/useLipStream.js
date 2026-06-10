import { useCallback, useRef } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// MediaPipe Face Mesh 입술 외곽 인덱스 (백엔드 data/lip_extractor.py와 일치)
const LIP_INDICES = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
];
const UPPER = 13, LOWER = 14, LEFT = 78, RIGHT = 308;
const ROI_SIZE = 112;
const TARGET_FPS = 25;
const PADDING = 0.25;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function cropLipROI(video, landmarks, ctx) {
  const w = video.videoWidth, h = video.videoHeight;
  let xs = [], ys = [];
  for (const i of LIP_INDICES) {
    xs.push(landmarks[i].x * w);
    ys.push(landmarks[i].y * h);
  }
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const bw = xMax - xMin, bh = yMax - yMin;
  const side = Math.max(bw * (1 + 2 * PADDING), bh * (1 + 2 * PADDING));
  const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2;
  const x0 = Math.max(0, Math.min(w - side, cx - side / 2));
  const y0 = Math.max(0, Math.min(h - side, cy - side / 2));

  ctx.drawImage(video, x0, y0, side, side, 0, 0, ROI_SIZE, ROI_SIZE);
  const img = ctx.getImageData(0, 0, ROI_SIZE, ROI_SIZE);
  // grayscale 변환 — 백엔드와 동일 (luminance)
  for (let i = 0; i < img.data.length; i += 4) {
    const g = 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
    img.data[i] = img.data[i + 1] = img.data[i + 2] = g;
  }
  ctx.putImageData(img, 0, 0);
}

export function useLipStream(videoRef, { onPartial, onStatus }) {
  const wsRef = useRef(null);
  const rafRef = useRef(null);
  const landmarkerRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const lastSendRef = useRef(0);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    onStatus?.('idle');
  }, [onStatus]);

  const start = useCallback(async () => {
    try {
      onStatus?.('loading');
      const fileset = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
      );
      landmarkerRef.current = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }, audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const canvas = document.createElement('canvas');
      canvas.width = ROI_SIZE; canvas.height = ROI_SIZE;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      canvasRef.current = canvas;

      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'partial') onPartial?.(msg.text);
      };
      ws.onopen = () => onStatus?.('streaming');
      ws.onerror = () => onStatus?.('error');
      ws.onclose = () => onStatus?.('idle');

      const interval = 1000 / TARGET_FPS;
      const tick = () => {
        if (!streamRef.current) return;
        const now = performance.now();
        if (now - lastSendRef.current >= interval && video.readyState >= 2) {
          lastSendRef.current = now;
          const result = landmarkerRef.current.detectForVideo(video, now);
          if (result.faceLandmarks?.[0]) {
            const lm = result.faceLandmarks[0];
            cropLipROI(video, lm, ctx);
            const mar = dist(lm[UPPER], lm[LOWER]) / (dist(lm[LEFT], lm[RIGHT]) + 1e-6);
            canvas.toBlob((blob) => {
              if (!blob || ws.readyState !== WebSocket.OPEN) return;
              const reader = new FileReader();
              reader.onload = () => {
                const b64 = String(reader.result).split(',')[1];
                ws.send(JSON.stringify({ type: 'frame', roi: b64, mar }));
              };
              reader.readAsDataURL(blob);
            }, 'image/png');
          }
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (err) {
      console.error(err);
      onStatus?.('error');
      stop();
    }
  }, [onPartial, onStatus, stop, videoRef]);

  return { start, stop };
}
