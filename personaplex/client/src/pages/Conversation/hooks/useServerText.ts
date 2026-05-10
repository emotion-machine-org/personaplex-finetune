import { useCallback, useEffect, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { useChatContext } from "../ChatContext";
import { decodeMessage } from "../../../protocol/encoder";

export const useServerText = () => {
  const [totalTextMessages, setTotalTextMessages] = useState(0);
  const { socket } = useSocketContext();
  const { addAssistantToken } = useChatContext();

  const onSocketMessage = useCallback((e: MessageEvent) => {
    const dataArray = new Uint8Array(e.data);
    const message = decodeMessage(dataArray);
    if (message.type === "text") {
      addAssistantToken(message.data);
      setTotalTextMessages((count) => count + 1);
    }
  }, [addAssistantToken]);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) return;
    currentSocket.addEventListener("message", onSocketMessage);
    return () => {
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket, onSocketMessage]);

  return { totalTextMessages };
};
