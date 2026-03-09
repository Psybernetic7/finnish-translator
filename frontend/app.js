/**
 * Finnish -> English Live Translator — client-side logic.
 *
 * Captures microphone audio via the Web Audio API (AudioWorklet), streams
 * raw PCM float32 samples at 16 kHz over a WebSocket to the FastAPI backend,
 * and displays the returned Finnish transcription + English translation.
 *
 * Audio pipeline:
 *   getUserMedia  ->  AudioContext (16 kHz)
 *                        -> MediaStreamSource
 *                        -> AudioWorkletNode (buffers & posts 100 ms chunks)
 *                        -> GainNode (gain=0, keeps graph alive without output)
 *                        -> destination
 *
 * Each 100 ms chunk (1 600 float32 samples) is sent as a binary WebSocket
 * frame. The server responds with JSON: { finnish: string, english: string }.
 */

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/**
 * WebSocket endpoint served by the FastAPI backend.
 *
 * Local dev:  'ws://localhost:8000/ws'
 * Modal:      'wss://<your-app>--serve.modal.run/ws'
 */
const WS_URL = 'ws://localhost:8000/ws';

// ---------------------------------------------------------------------------
// AudioWorklet processor (runs in a dedicated audio thread)
// ---------------------------------------------------------------------------

/**
 * Inlined AudioWorklet processor source.
 *
 * Loaded as a Blob URL so no separate file is needed. The processor collects
 * incoming PCM samples into a buffer and posts a Float32Array message to the
 * main thread every time 1 600 samples (100 ms at 16 kHz) have accumulated.
 */
const PROCESSOR_CODE = `
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._chunkSize = 1600; // 100 ms at 16 kHz
  }

  /**
   * Called by the audio rendering thread for each 128-sample render quantum.
   * Accumulates samples and posts 100 ms chunks to the main thread.
   */
  process(inputs) {
    const ch = inputs[0]?.[0]; // First input, first (mono) channel
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this._buf.push(ch[i]);
    while (this._buf.length >= this._chunkSize) {
      const chunk = new Float32Array(this._buf.splice(0, this._chunkSize));
      this.port.postMessage({ chunk }, [chunk.buffer]); // Transfer ownership
    }
    return true; // Keep processor alive
  }
}
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
`;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const micBtn    = document.getElementById('micBtn');
const statusEl  = document.getElementById('status');
const levelFill = document.getElementById('levelFill');
const finnishEl = document.getElementById('finnishText');
const englishEl = document.getElementById('englishText');
const clearBtn  = document.getElementById('clearBtn');

// ---------------------------------------------------------------------------
// Audio / WebSocket state
// ---------------------------------------------------------------------------

let ws          = null;   // WebSocket connection to the backend
let audioCtx    = null;   // AudioContext (16 kHz)
let workletNode = null;   // AudioWorkletNode running the capture processor
let sourceNode  = null;   // MediaStreamSourceNode (microphone input)
let gainNode    = null;   // GainNode (muted — keeps the audio graph alive)
let stream      = null;   // MediaStream from getUserMedia
let listening   = false;  // Whether we are currently recording

// ---------------------------------------------------------------------------
// Microphone capture & WebSocket streaming
// ---------------------------------------------------------------------------

/**
 * Request microphone access, set up the audio processing graph, and open
 * a WebSocket connection to the backend for streaming audio.
 */
async function startListening() {
  try {
    // Request microphone access (mono audio, no video)
    stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

    // Create an AudioContext at 16 kHz — the browser resamples automatically
    audioCtx = new AudioContext({ sampleRate: 16000 });

    // Load the AudioWorklet processor from an inline Blob URL
    const blob = new Blob([PROCESSOR_CODE], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    await audioCtx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    // Build the audio graph: mic -> worklet -> silent gain -> destination
    sourceNode  = audioCtx.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioCtx, 'audio-capture-processor');
    gainNode    = audioCtx.createGain();
    gainNode.gain.value = 0; // Mute output — we only need the worklet to capture
    sourceNode.connect(workletNode);
    workletNode.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    // Forward each audio chunk to the backend and update the level meter
    workletNode.port.onmessage = ({ data }) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(data.chunk.buffer); // Send raw PCM float32 as binary frame
      }
      // Compute RMS for the level meter animation
      let sum = 0;
      for (let i = 0; i < data.chunk.length; i++) sum += data.chunk[i] ** 2;
      const rms = Math.sqrt(sum / data.chunk.length);
      levelFill.style.width = Math.min(rms * 12, 1) * 100 + '%';
    };

    // Open a WebSocket connection to receive translation results
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen  = () => setStatus('Listening…', 'active');
    ws.onerror = () => setStatus('WebSocket error — is the server running?', 'error');
    ws.onclose = () => {
      if (listening) setStatus('Connection lost', 'error');
    };
    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      // Server sends { done: true } after flushing the final audio segment
      if (msg.done) {
        setStatus('Click the mic to start listening');
        ws?.close();
        ws = null;
        return;
      }
      appendText(finnishEl, msg.finnish);
      appendText(englishEl, msg.english);
    };

    listening = true;
    micBtn.classList.add('active');
  } catch (err) {
    setStatus('Error: ' + err.message, 'error');
    cleanup();
  }
}

/**
 * Stop recording: tear down audio immediately, then send a "flush" command
 * so the server processes any remaining audio. The WebSocket stays open
 * until the server replies with { done: true } (or a 60 s safety timeout).
 */
function stopListening() {
  listening = false;
  micBtn.classList.remove('active');
  levelFill.style.width = '0%';

  // Stop audio capture immediately
  workletNode?.disconnect();
  gainNode?.disconnect();
  sourceNode?.disconnect();
  audioCtx?.close();
  stream?.getTracks().forEach(t => t.stop());
  workletNode = gainNode = sourceNode = audioCtx = stream = null;

  // Ask the server to flush remaining audio, then wait for { done: true }
  if (ws && ws.readyState === WebSocket.OPEN) {
    setStatus('Processing remaining audio…');
    ws.send('flush');
    // Safety timeout — close even if the server doesn't respond
    setTimeout(() => {
      if (ws) {
        ws.close();
        ws = null;
        setStatus('Click the mic to start listening');
      }
    }, 60000);
  } else {
    ws?.close();
    ws = null;
    setStatus('Click the mic to start listening');
  }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

/**
 * Update the status label text and styling.
 *
 * @param {string} text    - Message to display.
 * @param {string} [type]  - CSS modifier class: 'active' (green) or 'error' (red).
 */
function setStatus(text, type = '') {
  statusEl.textContent = text;
  statusEl.className   = 'status ' + type;
}

/**
 * Append a new chunk of text to a transcript panel and auto-scroll.
 *
 * @param {HTMLElement} el   - The panel-text element.
 * @param {string}      text - Text to append.
 */
function appendText(el, text) {
  if (!text) return;
  const sep = el.textContent ? ' ' : '';
  el.textContent += sep + text;
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

/** Toggle recording on mic button click. */
micBtn.addEventListener('click', () => {
  if (listening) stopListening();
  else startListening();
});

/** Clear both transcript panels. */
clearBtn.addEventListener('click', () => {
  finnishEl.textContent = '';
  englishEl.textContent = '';
});
