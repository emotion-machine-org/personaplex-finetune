import { FC, useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../../SocketContext";
import { useUserAudio } from "../../hooks/useUserAudio";
import { ClientVisualizer } from "../AudioVisualizer/ClientVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";

type UserAudioProps = {
  theme: ThemeType;
};

export const UserAudio: FC<UserAudioProps> = ({ theme }) => {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const { sendMessage, socketStatus } = useSocketContext();
  const containerRef = useRef<HTMLDivElement>(null);

  const onRecordingStart = useCallback(() => {
    console.log("Recording started");
  }, []);

  const onRecordingStop = useCallback(() => {
    console.log("Recording stopped");
  }, []);

  const onRecordingChunk = useCallback(
    (chunk: Uint8Array) => {
      if (socketStatus !== "connected") return;
      sendMessage({ type: "audio", data: chunk });
    },
    [sendMessage, socketStatus],
  );

  const { startRecordingUser, stopRecording } = useUserAudio({
    constraints: {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    },
    onDataChunk: onRecordingChunk,
    onRecordingStart,
    onRecordingStop,
  });

  useEffect(() => {
    let res: Awaited<ReturnType<typeof startRecordingUser>>;
    if (socketStatus === "connected") {
      startRecordingUser().then((result) => {
        if (result) {
          res = result;
          setAnalyser(result.analyser);
        }
      });
    }
    return () => {
      console.log("Cleaning up recording + media stream.");
      stopRecording();
      res?.source?.disconnect();
      // Stop all MediaStream tracks so the mic is fully released.
      // Without this, the next getUserMedia (Model B) can fail on AirPods.
      res?.mediaStream?.getTracks().forEach((t) => t.stop());
    };
  }, [startRecordingUser, stopRecording, socketStatus]);

  return (
    <div className="user-audio w-32 h-32" ref={containerRef}>
      <ClientVisualizer theme={theme} analyser={analyser} parent={containerRef} />
    </div>
  );
};
