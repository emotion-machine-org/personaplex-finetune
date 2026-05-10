// @ts-nocheck
function asMs(samples) {
  return (samples * 1000 / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round(mili * sampleRate / 1000);
}

class MoshiProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    console.log("Moshi processor lives", currentFrame, sampleRate);
    console.log(currentTime);

    // Buffer length definitions
    let frameSize = asSamples(80);
    // initialBufferSamples: we wait to have at least that many samples before starting to play
    this.initialBufferSamples = 2 * frameSize;
    // once we have enough samples, we further wait that long before starting to play.
    // This allows to have buffer lengths that are not a multiple of frameSize.
    this.partialBufferSamples = asSamples(40);
    // If the buffer length goes over that many, we will drop the oldest packets until
    // we reach back initialBufferSamples + partialBufferSamples.
    // 300ms tolerance absorbs network jitter for cross-machine WebSocket connections.
    this.maxBufferSamples = asSamples(300);
    // increments
    this.partialBufferIncrement = asSamples(20);
    this.maxPartialWithIncrements = asSamples(160);
    this.maxBufferSamplesIncrement = asSamples(40);
    this.maxMaxBufferWithIncrements = asSamples(500);

    // State and metrics
    this.needsFadeIn = false;
    this.initState();

    this.silenceThreshold = 0.0;

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        console.log("Reset audio processor state.");
        this.initState();
        return;
      }
      if (event.data.type == "silenceThreshold") {
        this.silenceThreshold = event.data.value;
        return;
      }
      let frame = event.data.frame;
      this.frames.push(frame);
      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }
      if (this.pidx < 20) {
        console.log(this.timestamp(), "Got packet", this.pidx++, asMs(this.currentSamples()), asMs(frame.length))

      }
      if (this.currentSamples() >= this.totalMaxBufferSamples()) {
        console.log(this.timestamp(), "Dropping packets", asMs(this.currentSamples()), asMs(this.totalMaxBufferSamples()));
        let target = this.initialBufferSamples + this.partialBufferSamples;
        // Capture a crossfade tail from the audio we're about to skip
        let crossfadeSamples = Math.min(asSamples(5), this.currentSamples() - target);
        while (this.currentSamples() > target) {
          let first = this.frames[0];
          let to_remove = this.currentSamples() - target;
          to_remove = Math.min(first.length - this.offsetInFirstBuffer, to_remove);
          this.offsetInFirstBuffer += to_remove;
          this.timeInStream += to_remove / sampleRate;
          if (this.offsetInFirstBuffer == first.length) {
            this.frames.shift();
            this.offsetInFirstBuffer = 0;
          }
        }
        // Mark that we need a fade-in on the next output to smooth the skip
        this.needsFadeIn = true;
        console.log(this.timestamp(), "Packet dropped", asMs(this.currentSamples()));
        this.maxBufferSamples += this.maxBufferSamplesIncrement;
        this.maxBufferSamples = Math.min(this.maxMaxBufferWithIncrements, this.maxBufferSamples);
        console.log("Increased maxBuffer to", asMs(this.maxBufferSamples));
      }
      let delay = this.currentSamples() / sampleRate;
      this.port.postMessage({
        totalAudioPlayed: this.totalAudioPlayed,
        actualAudioPlayed: this.actualAudioPlayed,
        delay: event.data.micDuration - this.timeInStream,
        minDelay: this.minDelay,
        maxDelay: this.maxDelay,
      });
    };
  }

  initState() {
    this.frames = new Array();
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0.;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0.;
    this.actualAudioPlayed = 0.;
    this.maxDelay = 0.;
    this.minDelay = 2000.;
    // Debug
    this.pidx = 0;

    // Reset buffer params to defaults.
    this.partialBufferSamples = asSamples(40);
    this.maxBufferSamples = asSamples(300);
  }

  totalMaxBufferSamples() {
    return this.maxBufferSamples + this.partialBufferSamples + this.initialBufferSamples;
  }

  timestamp() {
    return Date.now() % 1000;
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  canPlay() {
    return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0;
  }

  process(inputs, outputs, parameters) {
    let delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }
    const output = outputs[0][0];
    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      return true;
    }
    if (this.firstOut) {
      console.log(this.timestamp(), "Audio resumed", asMs(this.currentSamples()), this.remainingPartialBufferSamples);
    }
    let first = this.frames[0];
    let out_idx = 0;
    while (out_idx < output.length && this.frames.length) {
      let first = this.frames[0];
      let to_copy = Math.min(first.length - this.offsetInFirstBuffer, output.length - out_idx);
      output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + to_copy), out_idx);
      this.offsetInFirstBuffer += to_copy;
      out_idx += to_copy;
      if (this.offsetInFirstBuffer == first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }
    if (this.firstOut) {
      this.firstOut = false;
      for (let i = 0; i < out_idx; i++) {
        output[i] *= i / out_idx;
      }
    }
    // Smooth fade-in after a packet drop skip to avoid clicks
    if (this.needsFadeIn && out_idx > 0) {
      this.needsFadeIn = false;
      let fadeLen = Math.min(out_idx, 64);
      for (let i = 0; i < fadeLen; i++) {
        output[i] *= i / fadeLen;
      }
    }
    if (out_idx < output.length) {
      // Buffer underrun — DON'T resetStart(). That forces a 200ms re-buffer delay
      // which causes the "spotty" audio pattern. Instead, just fade out gracefully
      // and keep playing as soon as the next packet arrives.
      if (!this._underrunCount) this._underrunCount = 0;
      if (this._underrunCount++ < 5) {
        console.log(this.timestamp(), "Underrun", output.length - out_idx, "samples short");
      }
      // Gentle fade-out on the partial audio we did play
      if (out_idx > 0) {
        let fadeLen = Math.min(out_idx, 64);
        for (let i = 0; i < fadeLen; i++) {
          output[out_idx - 1 - i] *= i / fadeLen;
        }
      }
      // Mark that next output should fade-in to avoid a click
      this.needsFadeIn = true;
    }
    // Silence gate: compute RMS and zero out if below threshold
    if (out_idx > 0) {
      let sumSq = 0;
      for (let i = 0; i < out_idx; i++) {
        sumSq += output[i] * output[i];
      }
      const rms = Math.sqrt(sumSq / out_idx);
      if (!this._rmsLogCount) this._rmsLogCount = 0;
      if (this._rmsLogCount++ % 200 === 0) {
        console.log("RMS:", rms.toFixed(6), "threshold:", this.silenceThreshold);
      }
      if (this.silenceThreshold > 0 && rms < this.silenceThreshold) {
        for (let i = 0; i < out_idx; i++) {
          output[i] = 0;
        }
      }
    }

    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    return true;
  }
}
registerProcessor("moshi-processor", MoshiProcessor);
