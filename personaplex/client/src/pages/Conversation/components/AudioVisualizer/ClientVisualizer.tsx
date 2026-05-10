import { FC, RefObject, useCallback, useEffect, useRef, useState } from "react";
import { clamp } from "../../hooks/audioUtils";
import { type ThemeType } from "../../hooks/useSystemTheme";

type AudioVisualizerProps = {
  analyser: AnalyserNode | null;
  parent: RefObject<HTMLElement>;
  theme: ThemeType;
};

const MAX_INTENSITY = 255;
const NUM_RINGS = 4;

const RING_COLORS = [
  "rgba(59, 130, 246, 0.85)",   // blue-500 core
  "rgba(59, 130, 246, 0.55)",
  "rgba(96, 165, 250, 0.35)",
  "rgba(147, 197, 253, 0.20)",
];

const RING_COLORS_IDLE = [
  "rgba(160, 160, 160, 0.20)",
  "rgba(160, 160, 160, 0.12)",
  "rgba(160, 160, 160, 0.07)",
  "rgba(160, 160, 160, 0.03)",
];

export const ClientVisualizer: FC<AudioVisualizerProps> = ({ analyser, parent, theme: _theme }) => {
  const [canvasWidth, setCanvasWidth] = useState(
    parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0
  );
  const requestRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothedIntensity = useRef(0);

  const draw = useCallback(
    (width: number, audioData: Uint8Array, ctx: CanvasRenderingContext2D) => {
      const cx = width / 2;
      const cy = width / 2;
      const maxRadius = width * 0.42;
      const baseRadius = width * 0.09;
      const ringThickness = width * 0.025;
      const ringGap = (maxRadius - baseRadius - ringThickness) / NUM_RINGS;

      const rms = Math.sqrt(
        audioData.reduce((acc, v) => acc + v * v, 0) / audioData.length
      );
      const raw = clamp(rms * 1.4, rms, MAX_INTENSITY) / MAX_INTENSITY;
      smoothedIntensity.current += (raw - smoothedIntensity.current) * 0.18;
      const intensity = smoothedIntensity.current;

      ctx.clearRect(0, 0, width, width);

      // Base circle
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius, 0, Math.PI * 2);
      ctx.fillStyle = intensity > 0.05 ? "rgba(59, 130, 246, 0.9)" : "rgba(160, 160, 160, 0.4)";
      ctx.fill();

      // Concentric rings
      for (let i = 0; i < NUM_RINGS; i++) {
        const threshold = (i + 1) / (NUM_RINGS + 1);
        const ringRadius = baseRadius + ringGap * (i + 1);
        const active = intensity > threshold * 0.6;

        ctx.beginPath();
        ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
        ctx.strokeStyle = active ? RING_COLORS[i] : RING_COLORS_IDLE[i];
        const thickness = active
          ? ringThickness * (0.6 + 0.8 * Math.min(1, (intensity - threshold * 0.6) / 0.3))
          : ringThickness * 0.4;
        ctx.lineWidth = thickness;
        ctx.stroke();
      }
    },
    [analyser]
  );

  const visualizeData = useCallback(() => {
    const width = parent.current
      ? Math.min(parent.current.clientWidth, parent.current.clientHeight)
      : 0;
    if (width !== canvasWidth) setCanvasWidth(width);
    requestRef.current = window.requestAnimationFrame(visualizeData);
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const audioData = new Uint8Array(140);
    analyser?.getByteFrequencyData(audioData);
    draw(width, audioData, ctx);
  }, [analyser, canvasWidth, parent, draw]);

  useEffect(() => {
    visualizeData();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [visualizeData, analyser]);

  return (
    <canvas
      ref={canvasRef}
      className="max-h-full max-w-full"
      width={canvasWidth}
      height={canvasWidth}
    />
  );
};
