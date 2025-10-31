from diffusers import StableDiffusionImg2ImgPipeline
import torch
from pathlib import Path

MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or "CompVis/stable-diffusion-v1-4" (requires license)
DEST = Path("../stable_diffusion/local_model").resolve()  # will create stable_diffusion/local_model

DEST.mkdir(parents=True, exist_ok=True)
print("Downloading model:", MODEL_ID)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16, safety_checker=None)
print("Saving to:", DEST)
pipe.save_pretrained(str(DEST))
print("Done")