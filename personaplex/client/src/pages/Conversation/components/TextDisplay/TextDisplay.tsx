import { FC, useEffect, useRef } from "react";
import { useServerText } from "../../hooks/useServerText";
import { useChatContext } from "../../ChatContext";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
};

export const TextDisplay: FC<TextDisplayProps> = ({ containerRef: _containerRef }) => {
  useServerText();
  const { messages } = useChatContext();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <span className="text-gray-300 text-sm">Waiting for response...</span>
      </div>
    );
  }

  return (
    <div className="h-full w-full max-h-full overflow-y-auto p-3 flex flex-col gap-2">
      {messages.map((msg) => (
        <div key={msg.id} className="flex justify-start">
          <div className="max-w-[85%] px-3 py-2 rounded-2xl rounded-bl-sm bg-gray-100 text-gray-800 text-sm leading-relaxed">
            {msg.text}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
};
