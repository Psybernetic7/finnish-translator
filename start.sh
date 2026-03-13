#!/bin/bash
# Finnish Translator — Vast.ai startup script
# Usage: bash start.sh
# Assumes a persistent volume mounted at /workspace

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Point HuggingFace cache to the persistent volume so models survive instance restarts
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
mkdir -p "$HF_HOME"

echo "==> Installing Python dependencies..."
pip install -q -r "$REPO_DIR/backend/requirements.txt"

echo "==> Starting server..."
cd "$REPO_DIR/backend"
uvicorn server:app --host 0.0.0.0 --port 8000
