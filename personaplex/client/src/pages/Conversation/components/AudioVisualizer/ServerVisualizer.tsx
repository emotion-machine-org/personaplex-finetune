import { FC, RefObject, useCallback, useEffect, useRef, useState } from "react";
import { clamp } from "../../hooks/audioUtils";
import { useSocketContext } from "../../SocketContext";
import { type ThemeType } from "../../hooks/useSystemTheme";

type AudioVisualizerProps = {
  analyser: AnalyserNode | null;
  parent: RefObject<HTMLElement>;
  theme: ThemeType;
};

const MAX_INTENSITY = 255;
const NUM_RINGS = 5;

// Color for each ring from inner to outer
const RING_COLORS = [
  "rgba(118, 185, 0, 0.9)",   // #76b900 core
  "rgba(118, 185, 0, 0.65)",
  "rgba(98, 161, 0, 0.45)",
  "rgba(78, 136, 0, 0.30)",
  "rgba(58, 111, 0, 0.18)",
];

const RING_COLORS_IDLE = [
  "rgba(160, 160, 160, 0.25)",
  "rgba(160, 160, 160, 0.15)",
  "rgba(160, 160, 160, 0.10)",
  "rgba(160, 160, 160, 0.06)",
  "rgba(160, 160, 160, 0.03)",
];

export const ServerVisualizer: FC<AudioVisualizerProps> = ({ analyser, parent, theme: _theme }) => {
  const [canvasWidth, setCanvasWidth] = useState(
    parent.current ? Math.min(parent.current.clientWidth, parent.current.clientHeight) : 0
  );
  const requestRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const smoothedIntensity = useRef(0);
  const { socketStatus } = useSocketContext();

  const draw = useCallback(
    (width: number, audioData: Uint8Array, ctx: CanvasRenderingContext2D) => {
      const cx = width / 2;
      const cy = width / 2;
      const maxRadius = width * 0.44;
      const baseRadius = width * 0.10;
      const ringThickness = width * 0.028;
      const ringGap = (maxRadius - baseRadius - ringThickness) / NUM_RINGS;

      // Compute RMS intensity
      const rms = Math.sqrt(
        audioData.reduce((acc, v) => acc + v * v, 0) / audioData.length
      );
      const raw = clamp(rms * 1.4, rms, MAX_INTENSITY) / MAX_INTENSITY;
      // Smooth it
      smoothedIntensity.current += (raw - smoothedIntensity.current) * 0.18;
      const intensity = smoothedIntensity.current;

      const connected = socketStatus === "connected";

      ctx.clearRect(0, 0, width, width);

      // Base circle
      ctx.beginPath();
      ctx.arc(cx, cy, baseRadius, 0, Math.PI * 2);
      ctx.fillStyle = connected ? "rgba(118, 185, 0, 1.0)" : "rgba(180, 180, 180, 0.5)";
      ctx.fill();

      // Concentric rings — each lights up based on intensity threshold
      for (let i = 0; i < NUM_RINGS; i++) {
        const threshold = (i + 1) / (NUM_RINGS + 1);
        const ringRadius = baseRadius + ringGap * (i + 1);
        const active = connected && intensity > threshold * 0.6;

        ctx.beginPath();
        ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
        ctx.strokeStyle = active ? RING_COLORS[i] : RING_COLORS_IDLE[i];
        // Scale ring thickness with intensity when active
        const thickness = active
          ? ringThickness * (0.6 + 0.8 * Math.min(1, (intensity - threshold * 0.6) / 0.3))
          : ringThickness * 0.5;
        ctx.lineWidth = thickness;
        ctx.stroke();
      }
    },
    [socketStatus]
  );

  const visualizeData = useCallback(() => {
    const width = parent.current
      ? Math.min(parent.current.clientWidth, parent.current.clientHeight)
      : 0;
    if (width !== canvasWidth) {
      setCanvasWidth(width);
    }
    requestRef.current = window.requestAnimationFrame(visualizeData);
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const audioData = new Uint8Array(140);
    analyser?.getByteFrequencyData(audioData);
    draw(width, audioData, ctx);
  }, [analyser, socketStatus, canvasWidth, parent, draw]);

  useEffect(() => {
    if (!analyser) return;
    analyser.smoothingTimeConstant = 0.95;
    visualizeData();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [visualizeData, analyser]);

  return (
    <canvas
      className="max-h-full max-w-full"
      ref={canvasRef}
      width={canvasWidth}
      height={canvasWidth}
    />
  );
};
