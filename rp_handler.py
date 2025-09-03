# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import sys
import json
import time
import traceback
import tempfile
import requests
import runpod
from typing import Dict, Any, Optional
import torch
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Wan modules
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import merge_video_audio, save_video

# Configuration
MODEL_NAME = "Wan-AI/Wan2.2-S2V-14B"
MAX_VIDEO_SIZE_MB = 100  # Maximum video size in MB
MAX_IMAGE_SIZE_MB = 20   # Maximum image size in MB
MAX_AUDIO_SIZE_MB = 50   # Maximum audio size in MB
DEFAULT_SIZE = "1024*704"
DEFAULT_FRAMES = 80
DEFAULT_SAMPLE_STEPS = 20

# Flash attention check
USE_FLASH_ATTN = bool(os.environ.get("USE_FLASH_ATTN", "1") not in ("0", "false", "False"))

def _has_flash_attn():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False

# Global model instance (loaded once)
wan_s2v_model = None

def load_model():
    """Load the Wan S2V model once at startup"""
    global wan_s2v_model
    
    logger.info(f"Loading Wan S2V model: {MODEL_NAME}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    use_flash = USE_FLASH_ATTN and _has_flash_attn()
    logger.info(f"Flash Attention enabled: {use_flash}")
    
    try:
        # Get model configuration
        cfg = WAN_CONFIGS["s2v-14B"]
        logger.info(f"Model config: {cfg}")
        
        # Load the Wan S2V model
        logger.info("Loading Wan S2V pipeline...")
        wan_s2v_model = wan.WanS2V(
            config=cfg,
            checkpoint_dir=MODEL_NAME,  # Will download from HuggingFace
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=True,  # Enable for memory efficiency
        )
        
        logger.info("Wan S2V model loaded successfully")
        
        # Warm up the model with a dummy run
        warmup_model()
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def warmup_model():
    """Warm up the model with a dummy inference"""
    try:
        logger.info("Warming up model...")
        
        # Create dummy files for warmup
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            # Create a simple test image
            dummy_img = Image.new('RGB', (512, 512), color='red')
            dummy_img.save(tmp_img.name)
            tmp_img_path = tmp_img.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            # Create a dummy audio file (1 second of silence)
            import numpy as np
            import soundfile as sf
            sample_rate = 16000
            duration = 1.0
            silence = np.zeros(int(sample_rate * duration))
            sf.write(tmp_audio.name, silence, sample_rate)
            tmp_audio_path = tmp_audio.name
        
        try:
            # Run a quick warmup generation
            with torch.no_grad():
                _ = wan_s2v_model.generate(
                    input_prompt="A cat sitting on a chair",
                    ref_image_path=tmp_img_path,
                    audio_path=tmp_audio_path,
                    num_repeat=1,
                    max_area=MAX_AREA_CONFIGS[DEFAULT_SIZE],
                    infer_frames=16,  # Small number for warmup
                    shift=1.0,
                    sample_solver='unipc',
                    sampling_steps=4,  # Very few steps for warmup
                    guide_scale=7.5,
                    seed=42,
                    offload_model=True,
                    init_first_frame=False,
                )
        except Exception as e:
            logger.warning(f"Warmup generation failed (non-critical): {str(e)}")
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_img_path):
                os.unlink(tmp_img_path)
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
                
        logger.info("Model warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {str(e)}")

def download_media(url: str, media_type: str = "auto", max_size_mb: int = None) -> bytes:
    """Download media file (image, audio, or video) with size limit"""
    if max_size_mb is None:
        if media_type == "video":
            max_size_mb = MAX_VIDEO_SIZE_MB
        elif media_type == "audio":
            max_size_mb = MAX_AUDIO_SIZE_MB
        else:
            max_size_mb = MAX_IMAGE_SIZE_MB
    
    logger.info(f"Downloading {media_type} from: {url}")
    
    try:
        # Stream download with size check
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get('Content-Length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValueError(f"{media_type.capitalize()} too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")
        
        # Download with size limit
        chunks = []
        total_size = 0
        max_size_bytes = max_size_mb * 1024 * 1024
        
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
            total_size += len(chunk)
            if total_size > max_size_bytes:
                raise ValueError(f"{media_type.capitalize()} exceeds maximum size of {max_size_mb}MB")
        
        media_data = b''.join(chunks)
        logger.info(f"Downloaded {total_size / (1024*1024):.2f}MB")
        return media_data
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download {media_type}: {str(e)}")
    except ValueError as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"Unexpected error downloading {media_type}: {str(e)}")

def generate_video(
    prompt: str,
    image_path: str,
    audio_path: str,
    size: str = DEFAULT_SIZE,
    infer_frames: int = DEFAULT_FRAMES,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    guide_scale: float = 7.5,
    seed: int = None
) -> str:
    """Generate video using Wan S2V model"""
    logger.info(f"Generating video with prompt: {prompt[:100]}...")
    logger.info(f"Image: {image_path}, Audio: {audio_path}")
    logger.info(f"Size: {size}, Frames: {infer_frames}, Steps: {sample_steps}")
    
    try:
        # Generate video
        video = wan_s2v_model.generate(
            input_prompt=prompt,
            ref_image_path=image_path,
            audio_path=audio_path,
            num_repeat=1,
            max_area=MAX_AREA_CONFIGS[size],
            infer_frames=infer_frames,
            shift=1.0,
            sample_solver='unipc',
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=seed if seed is not None else int(time.time()),
            offload_model=True,
            init_first_frame=False,
        )
        
        # Save video to temporary file
        timestamp = int(time.time())
        output_path = f"/tmp/generated_video_{timestamp}.mp4"
        
        logger.info(f"Saving generated video to {output_path}")
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=WAN_CONFIGS["s2v-14B"].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        # Merge video with audio
        logger.info("Merging video with audio...")
        merge_video_audio(video_path=output_path, audio_path=audio_path)
        
        # Clean up video tensor
        del video
        torch.cuda.synchronize()
        
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"Video generation failed: {str(e)}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler function for speech-to-video generation"""
    start_time = time.time()
    
    try:
        # Parse input
        job_input = job.get("input", {})
        prompt = job_input.get("prompt", "").strip()
        image_url = job_input.get("image_url", "").strip()
        audio_url = job_input.get("audio_url", "").strip()
        size = job_input.get("size", DEFAULT_SIZE)
        infer_frames = job_input.get("infer_frames", DEFAULT_FRAMES)
        sample_steps = job_input.get("sample_steps", DEFAULT_SAMPLE_STEPS)
        guide_scale = job_input.get("guide_scale", 7.5)
        seed = job_input.get("seed", None)
        
        # Validate input
        if not prompt:
            return {"error": "Missing 'prompt' in input"}
        if not image_url:
            return {"error": "Missing 'image_url' in input"}
        if not audio_url:
            return {"error": "Missing 'audio_url' in input"}
        
        # Validate parameters
        if size not in MAX_AREA_CONFIGS:
            return {"error": f"Unsupported size: {size}. Supported sizes: {list(MAX_AREA_CONFIGS.keys())}"}
        
        infer_frames = max(16, min(int(infer_frames), 120))  # Clamp between 16-120
        sample_steps = max(4, min(int(sample_steps), 50))    # Clamp between 4-50
        guide_scale = max(1.0, min(float(guide_scale), 20.0))  # Clamp between 1.0-20.0
        
        logger.info(f"Processing S2V request: prompt={prompt[:50]}...")
        logger.info(f"Image URL: {image_url}")
        logger.info(f"Audio URL: {audio_url}")
        logger.info(f"Parameters: size={size}, frames={infer_frames}, steps={sample_steps}")
        
        # Download media files
        image_data = download_media(image_url, "image")
        audio_data = download_media(audio_url, "audio")
        
        # Save to temporary files
        timestamp = int(time.time())
        image_path = f"/tmp/input_image_{timestamp}.jpg"
        audio_path = f"/tmp/input_audio_{timestamp}.wav"
        
        with open(image_path, "wb") as f:
            f.write(image_data)
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        try:
            # Generate video
            output_path = generate_video(
                prompt=prompt,
                image_path=image_path,
                audio_path=audio_path,
                size=size,
                infer_frames=infer_frames,
                sample_steps=sample_steps,
                guide_scale=guide_scale,
                seed=seed
            )
            
            # Read generated video
            with open(output_path, "rb") as f:
                video_data = f.read()
            
            # Encode video as base64 for response
            import base64
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Build metadata
            metadata = {
                "model": MODEL_NAME,
                "task": "s2v-14B",
                "size": size,
                "infer_frames": infer_frames,
                "sample_steps": sample_steps,
                "guide_scale": guide_scale,
                "processing_time_seconds": round(processing_time, 2),
                "video_size_mb": round(len(video_data) / (1024 * 1024), 2)
            }
            
            # Return success response
            return {
                "output": {
                    "video_base64": video_base64,
                    "metadata": metadata
                }
            }
            
        finally:
            # Clean up temporary files
            for temp_file in [image_path, audio_path, output_path]:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in handler: {error_msg}")
        traceback.print_exc()
        
        return {
            "error": error_msg,
            "details": traceback.format_exc() if os.environ.get("DEBUG") else None
        }

# Load model at startup
logger.info("Initializing Wan S2V 14B handler...")
load_model()

# Start RunPod serverless
if __name__ == "__main__":
    if os.environ.get("RUNPOD_POD_ID"):
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing
        logger.info("Running in local test mode...")
        
        # Test with example files
        test_job = {
            "input": {
                "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                "audio_url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
                "size": "1024*704",
                "infer_frames": 80,
                "sample_steps": 20,
                "guide_scale": 7.5
            }
        }
        
        logger.info("Testing S2V generation...")
        result = handler(test_job)
        logger.info("Test result:", json.dumps(result, indent=2))
