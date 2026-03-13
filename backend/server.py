"""
Finnish-to-English real-time speech translator — backend server.

Accepts streaming audio over a WebSocket, performs Finnish speech recognition
using faster-whisper, then translates the recognized text to English via
Helsinki-NLP's MarianMT model. Results are pushed back over the same WebSocket.

Architecture:
    Browser (PCM float32 @ 16 kHz)  --WebSocket-->  server
    server: accumulate audio buffer
            -> on silence or max duration, run ASR + translation in thread pool
            -> send {finnish, english} JSON back to client

Usage:
    Local:  uvicorn server:app
    Modal:  modal deploy modal_app.py
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Audio processing configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000             # Expected sample rate from the client (Hz)
SILENCE_THRESHOLD = 0.02        # RMS amplitude below which audio is "silent"
SILENCE_DURATION = 0.5          # Seconds of consecutive silence to trigger processing
MAX_BUFFER_SECONDS = 5          # Force processing after this many seconds even without silence
OVERLAP_SECONDS = 0.3           # Seconds of audio kept after processing for cross-segment context

# ---------------------------------------------------------------------------
# Model globals (loaded during startup via lifespan)
# ---------------------------------------------------------------------------
whisper_model = None
tokenizer = None
mt_model = None
mt_device = None

# Thread pool for running synchronous model inference without blocking the event loop
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, auto-detecting GPU if available."""
    global whisper_model, tokenizer, mt_model, mt_device

    import torch
    from faster_whisper import WhisperModel
    from transformers import MarianMTModel, MarianTokenizer

    # Auto-detect device: use GPU if available, otherwise CPU
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    mt_device = device

    # Try compute types in order of preference; some older GPUs don't support float16
    compute_types = ["float16", "int8_float16", "int8"] if use_cuda else ["int8"]
    whisper_model = None
    for compute_type in compute_types:
        try:
            print(f"Loading Whisper model (large-v3-turbo) on {device} ({compute_type})...")
            whisper_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
            break
        except ValueError as e:
            print(f"  {compute_type} not supported: {e}, trying next...")
    if whisper_model is None:
        raise RuntimeError("No supported compute type found for Whisper on this device.")

    print("Loading MarianMT fi->en model...")
    mt_model_name = "Helsinki-NLP/opus-mt-fi-en"
    tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
    mt_model = MarianMTModel.from_pretrained(mt_model_name).to(device)

    print(f"All models ready (device={device}).")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Finnish-to-English Live Translator API",
    description=(
        "Real-time Finnish speech recognition and translation service.\n\n"
        "## How it works\n\n"
        "1. Connect to the `/ws` WebSocket endpoint.\n"
        "2. Stream raw **PCM float32** audio frames (16 kHz, mono) as binary messages.\n"
        "3. The server buffers audio and automatically triggers processing when it "
        "detects a speech pause or the buffer reaches 5 seconds.\n"
        "4. Results are pushed back as JSON: `{\"finnish\": \"...\", \"english\": \"...\"}`.\n\n"
        "## Models\n\n"
        "| Task | Model | Source |\n"
        "|------|-------|--------|\n"
        "| Speech recognition | faster-whisper `large-v3-turbo` | [OpenAI Whisper](https://github.com/openai/whisper) |\n"
        "| Translation | MarianMT `opus-mt-fi-en` | [Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-fi-en) |\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sync_process(audio: np.ndarray) -> tuple[str | None, str | None]:
    """
    Run the full ASR + translation pipeline synchronously.

    Called inside the thread pool executor so it doesn't block the async loop.

    Args:
        audio: NumPy float32 array of raw PCM samples at 16 kHz.

    Returns:
        (finnish_text, english_translation) or (None, None) if no speech found.
    """
    # Step 1: Speech recognition — transcribe Finnish audio to text
    segments, _ = whisper_model.transcribe(
        audio,
        language="fi",
        beam_size=1,                                # Greedy decoding for speed
        vad_filter=True,                            # Filter out non-speech segments
        vad_parameters={"min_silence_duration_ms": 300},
        hallucination_silence_threshold=1.0,        # Skip segments likely hallucinated over silence
        no_speech_threshold=0.5,                    # Raise confidence bar for speech detection
    )
    # Filter out low-confidence segments (common Whisper hallucinations like "Kiitos")
    text = " ".join(
        s.text.strip() for s in segments
        if s.no_speech_prob < 0.5 and s.avg_logprob > -1.0
    ).strip()
    if not text:
        return None, None

    # Step 2: Machine translation — Finnish to English (on same device as model)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(mt_device) for k, v in inputs.items()}
    translated = mt_model.generate(**inputs, max_new_tokens=512)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return text, translation


def _is_silent(audio: np.ndarray) -> bool:
    """
    Check whether the tail of the audio buffer is silent.

    Computes the RMS (root mean square) of the last SILENCE_DURATION seconds
    and compares it against SILENCE_THRESHOLD.
    """
    silence_samples = int(SILENCE_DURATION * SAMPLE_RATE)
    if len(audio) < silence_samples:
        return False
    rms = float(np.sqrt(np.mean(audio[-silence_samples:] ** 2)))
    return rms < SILENCE_THRESHOLD


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Returns ok when the server and all models are loaded.",
)
async def health():
    """Simple health-check endpoint to verify the server is running and models are loaded."""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):  # noqa: D401
    """
    Streaming speech-translation WebSocket endpoint.

    Protocol:
        Client sends: raw PCM float32 binary frames (16 kHz, mono).
        Server sends: JSON messages {"finnish": "...", "english": "..."}.

    Audio is buffered and processed when either:
        - The trailing audio drops below the silence threshold, or
        - The buffer exceeds MAX_BUFFER_SECONDS.
    Processing is offloaded to a thread pool so the WebSocket can keep
    receiving audio during inference.
    """
    await ws.accept()
    print("WebSocket connected")
    loop = asyncio.get_running_loop()

    audio_buffer = np.array([], dtype=np.float32)
    overlap_samples = int(OVERLAP_SECONDS * SAMPLE_RATE)
    max_samples = int(MAX_BUFFER_SECONDS * SAMPLE_RATE)
    active_tasks: set[asyncio.Task] = set()
    ws_open = True

    async def process_and_send(audio: np.ndarray):
        """Run ASR + translation in a background thread, then send the result."""
        try:
            print(f"Processing {len(audio)/SAMPLE_RATE:.1f}s of audio...")
            finnish, english = await loop.run_in_executor(executor, _sync_process, audio)
            if finnish and english:
                print(f"Result: {finnish!r} -> {english!r}")
                if ws_open:
                    await ws.send_json({"finnish": finnish, "english": english})
                else:
                    print("WebSocket already closed — result dropped")
            else:
                print("No speech detected in segment")
        except Exception as e:
            print(f"Processing error: {e}")

    async def wait_for_tasks():
        """Wait for all in-flight processing tasks to complete."""
        if active_tasks:
            print(f"Waiting for {len(active_tasks)} in-flight task(s)...")
            await asyncio.gather(*active_tasks, return_exceptions=True)

    async def flush_remaining():
        """Wait for in-flight tasks, then process whatever audio is left in the buffer."""
        await wait_for_tasks()
        if len(audio_buffer) >= SAMPLE_RATE * 0.3:  # At least 0.3s
            finnish, english = await loop.run_in_executor(
                executor, _sync_process, audio_buffer
            )
            if finnish and english:
                print(f"Final result: {finnish!r} -> {english!r}")
                if ws_open:
                    await ws.send_json({"finnish": finnish, "english": english})

    try:
        while True:
            message = await ws.receive()

            # Handle client disconnect (raw ASGI message before Starlette raises)
            if message.get("type") == "websocket.disconnect":
                print("Client disconnected")
                break

            # Handle "flush" command: finish everything, signal done, then exit
            if "text" in message and message["text"] == "flush":
                print("Flush requested — processing remaining audio...")
                await flush_remaining()
                if ws_open:
                    await ws.send_json({"done": True})
                break

            # Normal path: binary audio data
            if "bytes" in message:
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                audio_buffer = np.concatenate([audio_buffer, chunk])

                # Clean up completed tasks
                active_tasks.difference_update({t for t in active_tasks if t.done()})

                # Decide whether to trigger processing
                should_process = (
                    not active_tasks
                    and len(audio_buffer) >= SAMPLE_RATE
                    and (_is_silent(audio_buffer) or len(audio_buffer) >= max_samples)
                )

                if should_process:
                    audio_to_process = audio_buffer.copy()
                    audio_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else np.array([], dtype=np.float32)
                    task = asyncio.create_task(process_and_send(audio_to_process))
                    active_tasks.add(task)

    except WebSocketDisconnect:
        print("WebSocket disconnected by client")
    except Exception as e:
        print(f"WebSocket error: {type(e).__name__}: {e}")
    finally:
        ws_open = False
        # Ensure any in-flight tasks finish (results won't be sent but won't crash)
        await wait_for_tasks()
