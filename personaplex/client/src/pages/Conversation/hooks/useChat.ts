import { useCallback, useRef, useState } from "react";
import { ChatMessage } from "../ChatContext";

let messageIdCounter = 0;

export const useChat = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const lastAssistantTokenTime = useRef(0);
  const interimId = useRef<number | null>(null);

  const addAssistantToken = useCallback((token: string) => {
    const now = Date.now();
    const gap = now - lastAssistantTokenTime.current;
    lastAssistantTokenTime.current = now;

    setMessages((prev) => {
      const lastNonInterim = [...prev].reverse().find((m) => !m.interim);
      const lastNonInterimIsUser = lastNonInterim?.role === "user";
      if (prev.length === 0 || gap > 800 || lastNonInterimIsUser) {
        return [...prev, { role: "assistant", text: token, id: ++messageIdCounter }];
      }
      // Find last assistant (non-interim) message to append to
      const updated = [...prev];
      for (let i = updated.length - 1; i >= 0; i--) {
        if (updated[i].role === "assistant" && !updated[i].interim) {
          updated[i] = { ...updated[i], text: updated[i].text + token };
          return updated;
        }
      }
      return [...prev, { role: "assistant", text: token, id: ++messageIdCounter }];
    });
  }, []);

  // Show interim user text immediately — creates or updates a single interim bubble
  const updateUserInterim = useCallback((text: string) => {
    if (!text.trim()) return;
    setMessages((prev) => {
      if (interimId.current !== null) {
        // Update existing interim bubble in place
        return prev.map((m) =>
          m.id === interimId.current ? { ...m, text: text.trim() } : m
        );
      }
      // Create new interim bubble
      const id = ++messageIdCounter;
      interimId.current = id;
      return [...prev, { role: "user", text: text.trim(), id, interim: true }];
    });
  }, []);

  // Finalize: replace interim with final text, or add new if no interim exists
  const finalizeUserMessage = useCallback((text: string) => {
    if (!text.trim()) return;
    setMessages((prev) => {
      if (interimId.current !== null) {
        const updated = prev.map((m) =>
          m.id === interimId.current
            ? { ...m, text: text.trim(), interim: false }
            : m
        );
        interimId.current = null;
        lastAssistantTokenTime.current = 0;
        return updated;
      }
      lastAssistantTokenTime.current = 0;
      return [...prev, { role: "user", text: text.trim(), id: ++messageIdCounter }];
    });
    interimId.current = null;
  }, []);

  const resetChat = useCallback(() => {
    setMessages([]);
    lastAssistantTokenTime.current = 0;
    interimId.current = null;
  }, []);

  return { messages, addAssistantToken, updateUserInterim, finalizeUserMessage, resetChat };
};
