# Multi-stage build for smaller image size
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in builder stage
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch and core dependencies
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.0+cu121 torchvision==0.20.0+cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    transformers==4.55.4 \
    accelerate>=0.33 \
    pillow>=10.4 \
    numpy>=2.0 \
    requests \
    einops \
    runpod==1.7.13 \
    soundfile \
    opencv-python-headless \
    diffusers>=0.31.0 \
    tokenizers>=0.20.3 \
    tqdm \
    easydict \
    ftfy \
    dashscope \
    hf_transfer \
    decord \
    safetensors \
    imageio-ffmpeg \
    imageio[ffmpeg]

# Final stage - minimal runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf_cache \
    HF_HUB_CACHE=/opt/hf_cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf_cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    USE_FLASH_ATTN=1

# Create cache directories
RUN mkdir -p /opt/hf_cache/hub

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# App code
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py
COPY wan/ /app/wan/

# Try to install optional dependencies
RUN pip install --no-cache-dir flash-attn==2.8.3 || echo "Flash attention not available, will use fallback"
RUN pip install --no-cache-dir decord || echo "Decord not available, video processing may be limited"

# Verify core installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
RUN python3 -c "import runpod; print('RunPod ready')"

CMD ["python3", "-u", "rp_handler.py"]
