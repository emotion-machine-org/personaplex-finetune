import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, useMemo, MutableRefObject } from "react";
import eruda from "eruda";
import { Conversation } from "../Conversation/Conversation";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const EVAL_PRESETS = [
  {
    label: "Emma (Companion)",
    category: "companion",
    text: "You are Emma, a friendly and supportive AI companion. Your role is to have natural, engaging conversations — listening actively, offering thoughtful responses, and helping the user think through whatever is on their mind. Be honest about what you do not know rather than guessing or making things up. You do not have access to day-to-day information such as what you did today, recent events in your own life, or personal experiences — if the user asks questions like these, gently redirect the conversation back to them. Be respectful, avoid offensive or hurtful language, and never pressure the user or make judgments about their choices.",
    greeting: "Hello there.",
    voice: "NATF0.pt",
  },
  {
    label: "Statin Check-in",
    category: "adherence",
    text: "PROGRAM: Lipitor Adherence Support\nPATIENT: Priya Sharma, 56F\nMEDICATION: Atorvastatin (Lipitor) 20mg daily, 8 weeks on treatment\nCLINICAL: Fill rate 100% through 8 weeks — two fills on schedule. No side effects reported. Baseline LDL was 188 mg/dL per referral notes. Patient was referred into program by prescribing cardiologist.\nPRIOR CONTACT: Enrollment call only — week 1. Brief, patient confirmed understanding of medication purpose. No issues raised.\nCALL GOAL: First routine check-in. Screen for side effects (especially muscle symptoms, which are common). Confirm continued adherence. Answer any questions about the program.\nFLAGS: None — patient appears adherent and engaged. Standard check-in protocol.",
    greeting: "Hi, is this Priya Sharma?",
    voice: "NATF1.pt",
  },
  {
    label: "Insurance — Marine Cargo",
    category: "insurance",
    text: "ROLE: Amanda Becker, Great Lakes Risk Partners, Marine & Cargo Insurance\nCLIENT: Folake Adeyemi, Facilities Manager, Meridian Logistics Group\nRELATIONSHIP: New client, ~8 months; professional rapport, building trust\nCALL GOAL: Quarterly policy review — discuss recent cargo claims trend, review coverage limits, and explore whether inland marine endorsement makes sense for their expanding warehouse network.",
    greeting: "Hi Folake, it's Amanda from Great Lakes Risk Partners. How are you doing today?",
    voice: "NATF2.pt",
  },
  {
    label: "Insurance — Cyber Liability",
    category: "insurance",
    text: "ROLE: Diane Takahashi, Meridian Risk Partners, cyber liability & tech E&O\nCLIENT: Folake Adeyemi, Facilities Manager, Caldwell Meridian Group\nRELATIONSHIP: Established, 3+ years; trusted advisor dynamic\nCALL GOAL: Proactive outreach — new SEC cyber disclosure rules affect the client's parent company. Discuss whether current cyber policy limits are adequate, and whether they need standalone coverage vs. endorsement.",
    greeting: "Hi Folake, it's Diane from Meridian Risk Partners. Do you have a few minutes?",
    voice: "NATF3.pt",
  },
  {
    label: "Insurance — Construction Bonds",
    category: "insurance",
    text: "ROLE: Jeffrey Abubakar, Meridian Construction Risk Partners, construction & surety bonds\nCLIENT: Rosa Antonelli, Facilities Manager, Crestfield Interiors Ltd.\nRELATIONSHIP: New client, ~8 months; still building trust\nCALL GOAL: Follow up on a recent bid bond request. Rosa's company is bidding on a large commercial renovation. Discuss bonding capacity, subcontractor requirements, and timeline.",
    greeting: "Hi Rosa, this is Jeffrey from Meridian Construction Risk. How's the bid coming along?",
    voice: "NATM0.pt",
  },
  {
    label: "Insurance — Personal Lines",
    category: "insurance",
    text: "ROLE: Stephanie Larsson, Brightway Northeast Insurance, personal lines generalist\nCLIENT: Haruto Suzuki, Fleet Manager, Caldwell Regional Logistics\nRELATIONSHIP: 3 years, comfortable and friendly\nCALL GOAL: Annual review of commercial auto fleet policy. Discuss recent accident history, new vehicle additions, and whether usage-based insurance makes sense for their delivery vans.",
    greeting: "Hey Haruto, it's Stephanie from Brightway. Got a minute for our annual review?",
    voice: "NATF1.pt",
  },
  {
    label: "Open-ended Chat",
    category: "companion",
    text: "You are a conversational partner. Have a natural, free-flowing conversation. Be curious, ask follow-up questions, and share thoughts when appropriate. Keep it casual and friendly.",
    greeting: "Hey, what's on your mind?",
    voice: "NATM1.pt",
  },
  {
    label: "Custom",
    category: "custom",
    text: "",
    greeting: "",
    voice: "NATF0.pt",
  },
];

// A/B model selection — same server, different query param
// "primary" = the main --moshi-weight model, "ab" = the --ab-moshi-weight model
const MODEL_QUERY: Record<string, string> = {
  base: "ab",
  finetuned: "primary",
  gemini: "gemini",
};

type ABPhase = "config" | "conv_a" | "conv_b" | "rate" | "submitted" | "quick_test";

type ABAssignment = {
  a: "base" | "finetuned";
  b: "base" | "finetuned";
};

function randomAssignment(): ABAssignment {
  return Math.random() < 0.5
    ? { a: "base", b: "finetuned" }
    : { a: "finetuned", b: "base" };
}

function getModelQuery(model: string): string {
  return MODEL_QUERY[model] || "primary";
}

export const Queue: FC = () => {
  const theme = "light" as const;
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState(false);
  const [conversationKey, setConversationKey] = useState(0);
  const [silenceThreshold] = useState(0.0);
  const modelParams = useModelParams();

  // A/B test state
  const [phase, setPhase] = useState<ABPhase>("config");
  const [assignment] = useState<ABAssignment>(randomAssignment);
  const [preference, setPreference] = useState<"a" | "b" | "tie" | null>(null);
  const [feedback, setFeedback] = useState("");
  const [testerId] = useState(() => `tester-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [quickTestModel, setQuickTestModel] = useState<string>("finetuned");
  const [selectedPreset, setSelectedPreset] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modelPath, setModelPath] = useState("");
  const [modelLoadStatus, setModelLoadStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [modelLoadMessage, setModelLoadMessage] = useState("");

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
      return () => { eruda.destroy(); };
    }
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, []);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
    }
    if (worklet.current) return;
    const ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, "moshi-processor");
    } catch {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, "moshi-processor");
    }
    worklet.current.connect(ctx.destination);
  }, []);

  const startConnection = useCallback(async () => {
    await startProcessor();
    await getMicrophoneAccess();
  }, [startProcessor, getMicrophoneAccess]);

  const startTest = useCallback(async () => {
    if (!hasMicrophoneAccess) {
      await startConnection();
    }
    setConversationKey((k) => k + 1);
    setPhase("conv_a");
  }, [hasMicrophoneAccess, startConnection]);

  const startQuickTest = useCallback(async () => {
    if (!hasMicrophoneAccess) {
      await startConnection();
    }
    setConversationKey((k) => k + 1);
    setPhase("quick_test");
  }, [hasMicrophoneAccess, startConnection]);

  const stopQuickTest = useCallback(() => {
    if (audioContext.current && worklet.current) {
      worklet.current.disconnect();
      worklet.current = new AudioWorkletNode(audioContext.current, "moshi-processor");
      worklet.current.connect(audioContext.current.destination);
    }
    setConversationKey((k) => k + 1);
    setPhase("config");
  }, []);

  const advanceToB = useCallback(() => {
    // Create a fresh worklet node so no stale audio from Model A leaks into Model B
    if (audioContext.current && worklet.current) {
      worklet.current.disconnect();
      worklet.current = new AudioWorkletNode(audioContext.current, "moshi-processor");
      worklet.current.connect(audioContext.current.destination);
    }
    setConversationKey((k) => k + 1);
    setPhase("conv_b");
  }, []);

  const advanceToRate = useCallback(() => {
    setPhase("rate");
  }, []);

  const submitRating = useCallback(async () => {
    const result = {
      timestamp: new Date().toISOString(),
      tester_id: testerId,
      assignment,
      preference,
      feedback: feedback.trim() || null,
      text_prompt: modelParams.textPrompt,
      voice_prompt: modelParams.voicePrompt,
      greeting: modelParams.greeting,
      params: {
        text_temperature: modelParams.textTemperature,
        text_topk: modelParams.textTopk,
        audio_temperature: modelParams.audioTemperature,
        audio_topk: modelParams.audioTopk,
        repetition_penalty: modelParams.repetitionPenalty,
      },
    };
    try {
      await fetch("/api/ab-result", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });
    } catch (e) {
      console.error("Failed to submit result:", e);
    }
    setPhase("submitted");
  }, [testerId, assignment, preference, feedback, modelParams]);

  const resetTest = useCallback(() => {
    setPhase("config");
    setPreference(null);
    setFeedback("");
    window.location.reload(); // re-randomize assignment
  }, []);

  const currentModelQuery = useMemo(() => {
    if (phase === "conv_a") return getModelQuery(assignment.a);
    if (phase === "conv_b") return getModelQuery(assignment.b);
    if (phase === "quick_test") return getModelQuery(quickTestModel);
    return "primary";
  }, [phase, assignment, quickTestModel]);

  const isReady = hasMicrophoneAccess && audioContext.current && worklet.current;
  const isConversation = phase === "conv_a" || phase === "conv_b" || phase === "quick_test";

  return (
    <div className="h-screen w-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="flex-none px-6 py-4 border-b border-gray-200 bg-white">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">PersonaPlex A/B Test</h1>
            <p className="text-xs text-gray-400">Compare two models — rate your preference</p>
          </div>
          {phase !== "config" && (
            <div className="flex items-center gap-3">
              <StepIndicator step={1} label="Model A" active={phase === "conv_a"} done={phase === "conv_b" || phase === "rate" || phase === "submitted"} />
              <div className="w-6 h-px bg-gray-300" />
              <StepIndicator step={2} label="Model B" active={phase === "conv_b"} done={phase === "rate" || phase === "submitted"} />
              <div className="w-6 h-px bg-gray-300" />
              <StepIndicator step={3} label="Rate" active={phase === "rate"} done={phase === "submitted"} />
            </div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden flex items-center justify-center">
        <div className="w-full max-w-3xl mx-auto px-6">

          {/* CONFIG PHASE */}
          {phase === "config" && (
            <div className="flex flex-col gap-4 py-6 overflow-y-auto max-h-[calc(100vh-100px)]">
              {/* Eval Preset Dropdown */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Evaluation Scenario</label>
                <select
                  value={selectedPreset}
                  onChange={(e) => {
                    const idx = Number(e.target.value);
                    setSelectedPreset(idx);
                    const p = EVAL_PRESETS[idx];
                    if (p.category !== "custom") {
                      modelParams.setTextPrompt(p.text);
                      modelParams.setGreeting(p.greeting);
                      modelParams.setVoicePrompt(p.voice);
                    }
                  }}
                  className="w-full p-2 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {EVAL_PRESETS.map((p, i) => (
                    <option key={i} value={i}>
                      {p.label}{p.category !== "custom" ? ` [${p.category}]` : ""}
                    </option>
                  ))}
                </select>
              </div>

              {/* System Prompt (always visible, editable) */}
              <div>
                <button
                  onClick={() => setPromptExpanded(!promptExpanded)}
                  className="text-xs text-gray-400 hover:text-gray-600 mb-1"
                >
                  {promptExpanded ? "▾ Hide system prompt" : "▸ Show/edit system prompt"}
                </button>
                {promptExpanded && (
                  <textarea
                    value={modelParams.textPrompt}
                    onChange={(e) => modelParams.setTextPrompt(e.target.value)}
                    className="w-full h-36 p-3 bg-white text-gray-800 text-sm border border-gray-200 rounded-lg resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                  />
                )}
              </div>

              {/* Greeting + Voice row */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Greeting</label>
                  <input
                    type="text"
                    value={modelParams.greeting}
                    onChange={(e) => modelParams.setGreeting(e.target.value)}
                    className="w-full p-2 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                    placeholder="Hello there."
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Voice</label>
                  <select
                    value={modelParams.voicePrompt}
                    onChange={(e) => modelParams.setVoicePrompt(e.target.value)}
                    className="w-full p-2 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                  >
                    {VOICE_OPTIONS.map((v) => (
                      <option key={v} value={v}>{v.replace(".pt", "")}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Advanced Params (collapsible) */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  {showAdvanced ? "▾ Hide parameters" : "▸ Sampling parameters"}
                </button>
                {showAdvanced && (
                  <div className="grid grid-cols-2 gap-3 mt-2">
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Text Temperature</label>
                      <input type="number" step="0.05" min="0.1" max="1.5"
                        value={modelParams.textTemperature}
                        onChange={(e) => modelParams.setTextTemperature(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Audio Temperature</label>
                      <input type="number" step="0.05" min="0.1" max="1.5"
                        value={modelParams.audioTemperature}
                        onChange={(e) => modelParams.setAudioTemperature(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Text Top-K</label>
                      <input type="number" step="5" min="1" max="500"
                        value={modelParams.textTopk}
                        onChange={(e) => modelParams.setTextTopk(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Audio Top-K</label>
                      <input type="number" step="10" min="1" max="500"
                        value={modelParams.audioTopk}
                        onChange={(e) => modelParams.setAudioTopk(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Repetition Penalty</label>
                      <input type="number" step="0.05" min="1.0" max="2.0"
                        value={modelParams.repetitionPenalty}
                        onChange={(e) => modelParams.setRepetitionPenalty(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-500 mb-0.5">Rep. Penalty Context</label>
                      <input type="number" step="8" min="0" max="200"
                        value={modelParams.repetitionPenaltyContext}
                        onChange={(e) => modelParams.setRepetitionPenaltyContext(Number(e.target.value))}
                        className="w-full p-1.5 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                      />
                    </div>
                  </div>
                )}
              </div>

              {showMicrophoneAccessMessage && (
                <p className="text-center text-red-500 text-sm">Please enable your microphone</p>
              )}

              {/* Model selector + action buttons */}
              <div className="flex items-center gap-2 justify-center">
                <span className="text-xs text-gray-400">Model:</span>
                {["finetuned", "base", "gemini"].map((m) => (
                  <button
                    key={m}
                    onClick={() => setQuickTestModel(m)}
                    className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                      quickTestModel === m
                        ? "border-gray-900 bg-gray-900 text-white"
                        : "border-gray-300 bg-white text-gray-600 hover:border-gray-400"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              <div className="flex gap-3">
                <button
                  onClick={startQuickTest}
                  className="flex-1 py-3 bg-gray-900 hover:bg-gray-800 text-white font-medium rounded-lg transition-colors text-sm"
                >
                  Quick Test
                </button>
                <button
                  onClick={startTest}
                  className="flex-1 py-3 bg-[#76b900] hover:bg-[#68a300] text-white font-medium rounded-lg transition-colors text-sm"
                >
                  A/B Test
                </button>
              </div>

              {/* Model swap */}
              <div className="border-t border-gray-200 pt-3 mt-1">
                <label className="block text-xs font-medium text-gray-600 mb-1">Load a different model (run path or .safetensors)</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={modelPath}
                    onChange={(e) => setModelPath(e.target.value)}
                    className="flex-1 p-2 bg-white text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#76b900] font-mono text-xs"
                    placeholder="/path/to/voice-training/runs/your_run_name"
                  />
                  <button
                    onClick={async () => {
                      if (!modelPath.trim()) return;
                      setModelLoadStatus("loading");
                      setModelLoadMessage("Loading model...");
                      try {
                        const resp = await fetch("/api/load-model", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ path: modelPath.trim() }),
                        });
                        const data = await resp.json();
                        if (resp.ok) {
                          setModelLoadStatus("success");
                          setModelLoadMessage(`Loaded: ${data.label}`);
                        } else {
                          setModelLoadStatus("error");
                          setModelLoadMessage(data.error || "Failed to load");
                        }
                      } catch (e) {
                        setModelLoadStatus("error");
                        setModelLoadMessage("Network error");
                      }
                    }}
                    disabled={modelLoadStatus === "loading" || !modelPath.trim()}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      modelLoadStatus === "loading"
                        ? "bg-gray-300 text-gray-500 cursor-wait"
                        : "bg-gray-900 hover:bg-gray-800 text-white"
                    }`}
                  >
                    {modelLoadStatus === "loading" ? "Loading..." : "Swap"}
                  </button>
                </div>
                {modelLoadMessage && (
                  <p className={`text-xs mt-1 ${
                    modelLoadStatus === "success" ? "text-green-600" : modelLoadStatus === "error" ? "text-red-500" : "text-gray-400"
                  }`}>
                    {modelLoadMessage}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* CONVERSATION PHASE */}
          {isConversation && isReady && (
            <div className="flex flex-col items-center gap-3 py-4 h-[calc(100vh-120px)]">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-900 text-white">
                  {phase === "quick_test" ? `Quick Test (${quickTestModel})` : phase === "conv_a" ? "Model A" : "Model B"}
                </span>
                {phase !== "quick_test" && (
                  <span className="text-xs text-gray-400">
                    {phase === "conv_a" ? "1 of 2" : "2 of 2"}
                  </span>
                )}
              </div>

              <div className="flex-1 w-full min-h-0">
                <Conversation
                  key={conversationKey}
                  workerAddr=""
                  modelQuery={currentModelQuery}
                  audioContext={audioContext as MutableRefObject<AudioContext | null>}
                  worklet={worklet as MutableRefObject<AudioWorkletNode | null>}
                  theme={theme}
                  startConnection={startConnection}
                  silenceThreshold={silenceThreshold}
                  {...modelParams}
                />
              </div>

              <button
                onClick={phase === "quick_test" ? stopQuickTest : phase === "conv_a" ? advanceToB : advanceToRate}
                className="flex-none px-6 py-2.5 bg-gray-900 hover:bg-gray-800 text-white font-medium rounded-lg transition-colors text-sm"
              >
                {phase === "quick_test" ? "Stop" : phase === "conv_a" ? "Stop → Next Model" : "Stop → Rate"}
              </button>
            </div>
          )}

          {/* RATING PHASE */}
          {phase === "rate" && (
            <div className="flex flex-col items-center gap-6 py-12">
              <h2 className="text-xl font-semibold text-gray-900">Which model did you prefer?</h2>
              <p className="text-sm text-gray-500 -mt-3">Consider naturalness, coherence, and voice quality</p>

              <div className="flex gap-3">
                {(["a", "b", "tie"] as const).map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setPreference(opt)}
                    className={`px-8 py-3 rounded-lg font-medium text-sm transition-all border-2 ${
                      preference === opt
                        ? "border-[#76b900] bg-[#76b900] text-white"
                        : "border-gray-200 bg-white text-gray-700 hover:border-gray-400"
                    }`}
                  >
                    {opt === "tie" ? "Tie" : `Model ${opt.toUpperCase()}`}
                  </button>
                ))}
              </div>

              <div className="w-full max-w-md">
                <label className="block text-xs font-medium text-gray-500 mb-1">Optional feedback</label>
                <textarea
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  className="w-full p-3 bg-white text-sm border border-gray-200 rounded-lg resize-none h-20 focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                  placeholder="What made you choose this? Any issues noticed?"
                />
              </div>

              <button
                onClick={submitRating}
                disabled={!preference}
                className={`px-8 py-3 rounded-lg font-medium text-sm transition-colors ${
                  preference
                    ? "bg-[#76b900] hover:bg-[#68a300] text-white"
                    : "bg-gray-200 text-gray-400 cursor-not-allowed"
                }`}
              >
                Submit Rating
              </button>
            </div>
          )}

          {/* SUBMITTED */}
          {phase === "submitted" && (
            <div className="flex flex-col items-center gap-4 py-12">
              <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" strokeWidth="2" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Thanks!</h2>
              <p className="text-sm text-gray-500">Your rating has been recorded.</p>
              <div className="text-xs text-gray-400 bg-gray-100 rounded-lg px-4 py-2">
                You preferred <strong>Model {preference === "tie" ? "—Tie" : preference?.toUpperCase()}</strong>
                {" "}(which was the <strong>{preference === "tie" ? "n/a" : assignment[preference!]}</strong> model)
              </div>
              <button
                onClick={resetTest}
                className="mt-4 px-6 py-2.5 bg-gray-900 hover:bg-gray-800 text-white font-medium rounded-lg transition-colors text-sm"
              >
                New Test
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Step indicator pill
const StepIndicator: FC<{ step: number; label: string; active: boolean; done: boolean }> = ({
  step, label, active, done,
}) => (
  <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
    active ? "bg-gray-900 text-white" : done ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-400"
  }`}>
    <span>{done ? "✓" : step}</span>
    <span>{label}</span>
  </div>
);
