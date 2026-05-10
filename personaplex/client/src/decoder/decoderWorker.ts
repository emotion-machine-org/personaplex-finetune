// Factory function to create a decoder worker
const createWorkerInstance = (): Worker => {
  const worker = new Worker(
    new URL("/assets/decoderWorker.min.js", import.meta.url),
  );
  worker.onerror = (event) => {
    console.error("Decoder worker error:", event.message);
  };
  return worker;
};

// Send init command to a worker (no fake BOS page — let the real Opus stream initialize the decoder cleanly)
const sendInitCommand = (worker: Worker, audioContextSampleRate: number): void => {
  worker.postMessage({
    command: "init",
    bufferLength: 960 * audioContextSampleRate / 24000,
    decoderSampleRate: 24000,
    outputBufferSampleRate: audioContextSampleRate,
    resampleQuality: 5,
  });
};

// Factory function to create a fresh decoder worker
export const createDecoderWorker = (): Worker => {
  return createWorkerInstance();
};

// Initialize a decoder worker and return a promise that resolves when ready
export const initDecoder = (worker: Worker, audioContextSampleRate: number): Promise<void> => {
  return new Promise((resolve) => {
    console.log("Starting decoder initialization");
    sendInitCommand(worker, audioContextSampleRate);
    // WASM loads fast — 300ms is plenty
    setTimeout(() => {
      console.log("Decoder initialization complete");
      resolve();
    }, 300);
  });
};
