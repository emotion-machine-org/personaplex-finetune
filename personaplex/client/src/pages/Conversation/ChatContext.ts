import { createContext, useContext } from "react";

export type ChatMessage = {
  role: "assistant" | "user";
  text: string;
  id: number;
  interim?: boolean;
};

export type ChatContextType = {
  messages: ChatMessage[];
  addAssistantToken: (token: string) => void;
  updateUserInterim: (text: string) => void;
  finalizeUserMessage: (text: string) => void;
};

export const ChatContext = createContext<ChatContextType | null>(null);

export const useChatContext = () => {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used within ChatContext.Provider");
  return ctx;
};
