from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np

def load_stable_diffusion_model(local_dir="stable_diffusion/local_model"):
    """
    Load a diffusers pipeline from a local folder.
    - Chooses dtype based on available device (float16 on CUDA, float32 on CPU).
    - Enables attention slicing / VAE tiling to reduce peak memory.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pick dtype safely
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # try img2img first, fall back to text2img
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(local_dir, torch_dtype=torch_dtype)
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(local_dir, torch_dtype=torch_dtype)

    # Move to device (float16 pipelines must stay on cuda)
    try:
        pipe = pipe.to(device)
    except Exception:
        # some objects may not support .to or moving fails; keep pipe as-is
        pass

    # Enable memory-saving features (safe to call when available)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        # optional beneficial call if available
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        # xformers can speed up/ reduce memory if installed
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe

def _ensure_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    try:
        import torch as _torch
        if isinstance(image, _torch.Tensor):
            arr = image.cpu().permute(1, 2, 0).numpy()
            # assume 0..1 or 0..255
            if arr.max() <= 1.1:
                arr = (arr * 255).astype("uint8")
            else:
                arr = arr.astype("uint8")
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        pass
    raise ValueError("Input is in incorrect format. Currently, we only support PIL.Image, numpy.ndarray, or torch.Tensor")

def generate_image_from_prompt(pipe, prompt, init_image=None, strength=0.75, guidance_scale=7.5, num_inference_steps=15):
    """
    If init_image provided and pipeline is img2img, runs img2img; otherwise runs text2img.
    Returns a PIL.Image.
    - Uses a smaller default step count to reduce CPU time.
    - Resizes init image (max side 512) to meet pipeline expectations.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipe = pipe.to(device)
    except Exception:
        pass

    if init_image is not None:
        init_image = _ensure_pil(init_image)
        # Resize to max 512 while preserving aspect ratio (img2img expects reasonable sizes)
        max_dim = 512
        w, h = init_image.size
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            new_size = (int(w * ratio), int(h * ratio))
            init_image = init_image.resize(new_size, resample=Image.LANCZOS)

    # If pipeline is an Img2Img pipeline, require init_image
    if isinstance(pipe, StableDiffusionImg2ImgPipeline):
        if init_image is None:
            raise ValueError("Pipeline expects an init_image (img2img). Provide an uploaded image.")
        out = pipe(prompt=prompt, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
    else:
        out = pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)

    img = out.images[0]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))
    return img