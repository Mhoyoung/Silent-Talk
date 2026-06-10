import React from 'react';

export default function WebcamView({ videoRef, onStart, onStop, status }) {
  const isOn = status === 'streaming';
  return (
    <section className="rounded-2xl bg-slate-900/60 border border-slate-800 p-4 flex flex-col">
      <h2 className="text-sm font-medium mb-3 text-slate-300">웹캠</h2>
      <div className="relative flex-1 rounded-xl overflow-hidden bg-black aspect-video">
        <video
          ref={videoRef}
          className="w-full h-full object-cover scale-x-[-1]"
          autoPlay
          playsInline
          muted
        />
        {!isOn && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-500">
            "시작" 버튼을 눌러 웹캠을 활성화하세요
          </div>
        )}
      </div>
      <div className="mt-3 flex gap-2">
        <button
          onClick={onStart}
          disabled={isOn}
          className="px-4 py-2 rounded-lg bg-accent text-ink font-medium disabled:opacity-40"
        >
          시작
        </button>
        <button
          onClick={onStop}
          disabled={!isOn}
          className="px-4 py-2 rounded-lg bg-slate-700 text-slate-100 disabled:opacity-40"
        >
          정지
        </button>
      </div>
    </section>
  );
}
