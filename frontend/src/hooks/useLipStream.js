import { useCallback, useRef } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { FPS, ROI_SIZE } from '../constants.js';
import { cropLipROI, mouthRatio } from '../webcamUtils.js';

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

      const interval = 1000 / FPS;
      const tick = () => {
        if (!streamRef.current) return;
        const now = performance.now();
        if (now - lastSendRef.current >= interval && video.readyState >= 2) {
          lastSendRef.current = now;
          const result = landmarkerRef.current.detectForVideo(video, now);
          if (result.faceLandmarks?.[0]) {
            const lm = result.faceLandmarks[0];
            cropLipROI(video, lm, ctx);
            const mar = mouthRatio(lm);
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
