import React, { useEffect, useRef } from 'react';

export default function CaptionPanel({ captions }) {
  const endRef = useRef(null);
  useEffect(() => endRef.current?.scrollIntoView({ behavior: 'smooth' }), [captions]);

  return (
    <section className="rounded-2xl bg-slate-900/60 border border-slate-800 p-4 flex flex-col">
      <h2 className="text-sm font-medium mb-3 text-slate-300">실시간 자막</h2>
      <div className="flex-1 overflow-y-auto space-y-2 pr-2">
        {captions.length === 0 ? (
          <p className="text-slate-500 text-sm">발화가 감지되면 여기에 표시됩니다.</p>
        ) : (
          captions.map((c) => (
            <div
              key={c.id}
              className="rounded-lg bg-slate-800/60 border border-slate-700 px-3 py-2 text-slate-100"
            >
              {c.text}
            </div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </section>
  );
}
