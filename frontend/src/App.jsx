import React, { useEffect, useRef, useState } from 'react';
import WebcamView from './components/WebcamView.jsx';
import CaptionPanel from './components/CaptionPanel.jsx';
import { useLipStream } from './hooks/useLipStream.js';

export default function App() {
  const videoRef = useRef(null);
  const [captions, setCaptions] = useState([]);
  const [status, setStatus] = useState('idle');

  const { start, stop } = useLipStream(videoRef, {
    onPartial: (text) => {
      if (!text) return;
      setCaptions((prev) => [...prev.slice(-9), { id: Date.now(), text }]);
    },
    onStatus: setStatus,
  });

  useEffect(() => () => stop(), [stop]);

  return (
    <div className="min-h-full flex flex-col">
      <header className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
        <h1 className="text-xl font-semibold">
          Silent Talk <span className="text-accent">·</span>{' '}
          <span className="text-slate-400 text-sm font-normal">실시간 립리딩</span>
        </h1>
        <span
          className={`text-xs px-2 py-1 rounded ${
            status === 'streaming' ? 'bg-emerald-500/20 text-emerald-300'
            : status === 'error' ? 'bg-red-500/20 text-red-300'
            : 'bg-slate-700/50 text-slate-300'
          }`}
        >
          {status}
        </span>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 p-6">
        <WebcamView videoRef={videoRef} onStart={start} onStop={stop} status={status} />
        <CaptionPanel captions={captions} />
      </main>

      <footer className="px-6 py-3 text-xs text-slate-500 border-t border-slate-800">
        1차 MVP — 어휘 100 / 문장 50. 모델: 3D-CNN + Bi-GRU + CTC.
      </footer>
    </div>
  );
}
