import { useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { useMediaContext } from "../MediaContext";

const SERVER_SAMPLE_RATE = 24000;

export type AudioStats = {
  playedAudioDuration: number;
  missedAudioDuration: number;
  totalAudioMessages: number;
  delay: number;
  minPlaybackDelay: number;
  maxPlaybackDelay: number;
};

type useServerAudioArgs = {
  setGetAudioStats?: (getAudioStats: () => AudioStats) => void;
};

type WorkletStats = {
  totalAudioPlayed: number;
  actualAudioPlayed: number;
  delay: number;
  minDelay: number;
  maxDelay: number;
};

export const useServerAudio = ({setGetAudioStats}: useServerAudioArgs) => {
  const { socket, socketStatus } = useSocketContext();
  const {startRecording, stopRecording, audioContext, worklet, micDuration, actualAudioPlayed } =
    useMediaContext();
  const analyser = useRef(audioContext.current.createAnalyser());
  worklet.current.connect(analyser.current);
  const startTime = useRef<number | null>(null);
  const [hasCriticalDelay, setHasCriticalDelay] = useState(false);
  const totalAudioMessages = useRef(0);
  const receivedDuration = useRef(0);
  const workletStats = useRef<WorkletStats>({
    totalAudioPlayed: 0,
    actualAudioPlayed: 0,
    delay: 0,
    minDelay: 0,
    maxDelay: 0,});

  const onWorkletMessage = useCallback(
    (event: MessageEvent<WorkletStats>) => {
      workletStats.current = event.data;
      actualAudioPlayed.current = workletStats.current.actualAudioPlayed;
    },
    [],
  );
  worklet.current.port.onmessage = onWorkletMessage;

  const getAudioStats = useCallback(() => {
    return {
      playedAudioDuration: workletStats.current.actualAudioPlayed,
      delay: workletStats.current.delay,
      minPlaybackDelay: workletStats.current.minDelay,
      maxPlaybackDelay: workletStats.current.maxDelay,
      missedAudioDuration: workletStats.current.totalAudioPlayed - workletStats.current.actualAudioPlayed,
      totalAudioMessages: totalAudioMessages.current,
    };
  }, []);

  // Resample ratio (computed once)
  const resampleRatio = audioContext.current.sampleRate / SERVER_SAMPLE_RATE;

  let midx = 0;
  const decodeAudio = useCallback((data: Uint8Array) => {
    // Raw int16 PCM at 24kHz from server — no Opus decode needed
    if (data.length < 2) return;

    if (midx < 5) {
      console.log(Date.now() % 1000, "Got RAW PCM",
        midx++,
        "bytes:", data.length,
        "samples:", data.length / 2);
    }

    // Convert int16 → float32
    const int16 = new Int16Array(data.buffer, data.byteOffset, data.length >> 1);
    const srcLen = int16.length;

    // Resample 24kHz → AudioContext rate (typically 48kHz) via linear interpolation
    const outLen = Math.round(srcLen * resampleRatio);
    const float32 = new Float32Array(outLen);

    if (resampleRatio === 2) {
      // Fast path for exact 2x (24→48kHz)
      for (let i = 0; i < srcLen; i++) {
        const s = int16[i] / 32768.0;
        const nextS = i + 1 < srcLen ? int16[i + 1] / 32768.0 : s;
        float32[i * 2] = s;
        float32[i * 2 + 1] = (s + nextS) * 0.5;
      }
    } else {
      // Generic linear interpolation
      for (let i = 0; i < outLen; i++) {
        const srcIdx = i / resampleRatio;
        const left = Math.floor(srcIdx);
        const right = Math.min(left + 1, srcLen - 1);
        const frac = srcIdx - left;
        float32[i] = (int16[left] * (1 - frac) + int16[right] * frac) / 32768.0;
      }
    }

    receivedDuration.current += float32.length / audioContext.current.sampleRate;
    worklet.current.port.postMessage({frame: float32, type: "audio", micDuration: micDuration.current});
  }, []);

  const onSocketMessage = useCallback(
    (e: MessageEvent) => {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);
      if (message.type === "audio") {
        decodeAudio(message.data);
        totalAudioMessages.current++;
      }
    },
    [decodeAudio],
  );

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket || socketStatus !== "connected") {
      return;
    }
    worklet.current.port.postMessage({type: "reset"});
    console.log(Date.now() % 1000, "Audio ready — raw PCM mode (no Opus decoder)");
    startRecording();
    currentSocket.addEventListener("message", onSocketMessage);
    totalAudioMessages.current = 0;
    return () => {
      console.log("Stop recording called in cleanup.")
      stopRecording();
      startTime.current = null;
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket, socketStatus]);

  useEffect(() => {
    if (setGetAudioStats) {
      setGetAudioStats(getAudioStats);
    }
  }, [setGetAudioStats, getAudioStats]);

  return {
    decodeAudio,
    analyser,
    getAudioStats,
    hasCriticalDelay,
    setHasCriticalDelay,
  };
};
