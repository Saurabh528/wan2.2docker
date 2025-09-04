# CUDA 12.1 runtime + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    HF_HOME=/opt/hf_cache \
    HF_HUB_CACHE=/opt/hf_cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1

# Create cache directories - for storing models
RUN mkdir -p /opt/hf_cache/hub

# System deps (Python 3.10 is default on 22.04)
# Add all build tools + FFmpeg *dev* headers so eva-decord can compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl ca-certificates \
    ffmpeg pkg-config \
    libsm6 libxext6 libgl1 \
    build-essential cmake ninja-build \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# Make sure pip is up-to-date for this Python
RUN python3 -m pip install --upgrade pip setuptools wheel


# --- Install hf_transfer for fast downloads ---
RUN pip install --no-cache-dir hf_transfer

# --- High-level libs (kept exactly as you listed) ---
RUN pip install --no-cache-dir \
    transformers==4.55.4 \
    "accelerate>=0.33" \
    "pillow>=10.4" \
    "numpy>=2.0" \
    runpod==1.7.13 \
    eva-decord==0.6.1 \
    librosa \
    requests \
    einops \
    timm \
    huggingface_hub \
    soundfile \
    opencv-python-headless \
    "diffusers>=0.31.0" \
    "tokenizers>=0.20.3" \
    tqdm \
    easydict \
    ftfy \
    dashscope \
    safetensors \
    imageio-ffmpeg \
    imageio[ffmpeg]

# --- Install eva-decord on Ubuntu 22.04 + Python 3.10: build from source ---
# There are no Linux cp310 wheels on PyPI, so we force a source build.

# Optional: FlashAttention (skip gracefully if no matching wheel)

# 2. Install stable Colab stack
RUN pip install --no-cache-dir torch==2.2.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash-attn==2.5.6 
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu121


# --- App code ---
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py
COPY wan/ /app/wan/

# --- Package verification (prints versions so you *know* it's there) ---
RUN python3 -c "import sys; print('Python:', sys.version)"
RUN python3 -c "import torch, torchvision, torchaudio; print('Torch OK:', torch.__version__); print('TV:', torchvision.__version__); print('TA:', torchaudio.__version__)"
RUN python3 -c "import transformers; print('Transformers:', transformers.__version__)"
RUN python3 -c "import einops, timm, hf_transfer; print('einops OK:', einops.__version__); print('timm OK:', timm.__version__); print('hf_transfer OK')"
RUN python3 -c "import decord; print('eva_decord OK:', getattr(decord, '__version__', 'unknown'))"

# --- Default command ---
CMD ["python3", "-u", "rp_handler.py"]
