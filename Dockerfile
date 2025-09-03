# CUDA 12.1 runtime + Ubuntu 22.04 (optimized for RunPod)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/opt/hf_cache \
    HF_HOME=/opt/hf_cache \
    HF_HUB_CACHE=/opt/hf_cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    USE_FLASH_ATTN=1

# Create cache directories for storing models
RUN mkdir -p /opt/hf_cache/hub

# System deps (python3.10 is default on 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget ffmpeg libsm6 libxext6 libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip/setuptools/wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# --- Core CUDA-matched PyTorch stack (cu121 wheels) ---
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121

# --- Install hf_transfer for fast downloads ---
RUN pip install --no-cache-dir hf_transfer

# --- High-level libs (latest compatible) ---
RUN pip install --no-cache-dir \
    transformers==4.55.4 \
    accelerate>=0.33 \
    pillow>=10.4 \
    numpy>=2.0 \
    requests \
    einops \
    timm \
    runpod==1.7.13 \
    soundfile \
    imageio[ffmpeg] \
    opencv-python-headless \
    diffusers>=0.31.0 \
    tokenizers>=0.20.3 \
    tqdm \
    easydict \
    ftfy \
    dashscope

# Install FlashAttention (fast-path). If no wheel matches- skip it.
RUN pip install --no-cache-dir flash-attn==2.8.3 || echo "Flash attention installation failed, will use fallback"

# App code
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py
COPY wan/ /app/wan/
COPY examples/ /app/examples/

# Verify package installations
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python3 -c "import transformers; print('Transformers cache initialized')"
RUN python3 -c "import runpod; print('RunPod installed')"
RUN python3 -c "import soundfile; print('SoundFile installed')"

# Test flash attention availability
RUN python3 -c "import flash_attn; print('Flash Attention available')" 2>/dev/null || echo "Flash Attention not available, will use fallback"

CMD ["python3", "-u", "rp_handler.py"]
