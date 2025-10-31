# DeepFake AI System

A Streamlit app that performs deepfake detection (VAE + ViT) and image generation (GAN wrapper + Stable Diffusion img2img).  
This README covers file layout, prerequisites, how to run locally, where to put model files, and common troubleshooting.

## File layout (important files)
- app.py — main Streamlit application (UI + logic)
- stable_diffusion/sd_utils.py — Stable Diffusion helper functions (loader + img2img)
- progan_generator_final.pt — GAN checkpoint (place in project root or upload via sidebar)
- vae_model.pth, best_vit_deepfake_detector.pt — detector checkpoints (optional)
- stable_diffusion/local_model/ — local diffusers pipeline folder (required for Stable Diffusion img2img)
- requirements.txt — Python package dependencies (see below)

## Prerequisites
- Python 3.8–3.11
- Git (optional)
- Recommended: a virtual environment (venv or conda)
- If you want to use GPU acceleration, install a matching PyTorch build for your CUDA version.

## Setup (Windows PowerShell)
1. Create & activate a venv (recommended)
   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Install PyTorch (choose the correct command from https://pytorch.org/get-started/locally).
   - Example (CPU-only):
     ```
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```
   - Example (CUDA — replace cu118 with your CUDA version):
     ```
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

3. Install the rest of the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) To reduce TensorFlow oneDNN informational messages:
   ```
   $env:TF_ENABLE_ONEDNN_OPTS = "0"
   ```

## Running the app
From the project root (where `app.py` is located):
```
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
Open the URL shown by Streamlit in your browser (usually http://localhost:8501).

## Model files — where to put them
- GAN checkpoint (`progan_generator_final.pt`)
  - Place into the project root (same folder as `app.py`) OR into `models/` subfolder.
  - Alternatively you can upload the GAN checkpoint via the app sidebar — the app will save it as `progan_generator_final.pt` into the project folder.
- Detector checkpoints
  - `vae_model.pth` and `best_vit_deepfake_detector.pt` — place in the project root or `models/`.
- Stable Diffusion (img2img)
  - Must be a local diffusers pipeline saved to `stable_diffusion/local_model/` (must contain config.json, unet, vae, text_encoder/tokenizer files).
  - If you want a one-time download and save, create the pipeline in a separate script using diffusers and call `pipe.save_pretrained("stable_diffusion/local_model")`. The app will not auto-download models.

## Typical usage flow
1. Start the app.
2. In the sidebar:
   - Upload GAN checkpoint (optional) or copy it to project root.
   - Click "🔄 Load Checkpoint Models" to load VAE / ViT / GAN wrappers.
   - Click "🔄 Load Stable Diffusion Pipeline" to load the local SD pipeline (one-time per Streamlit session).
3. Detection tab:
   - Upload an input image and click "🔍 Analyze Image".
4. Generation tab:
   - Upload an image, select "GAN (state)" (uses `progan_generator_final.pt`) or "Diffusion (img2img)" (uses loaded SD pipeline), then click "🎨 Generate".

## Troubleshooting
- SciPy missing errors: `pip install scipy`
- If Stable Diffusion load hangs or prints "Loading pipeline components..." — ensure `stable_diffusion/local_model` exists and contains a saved diffusers pipeline. The loader uses `local_files_only=True` and won't download model files.
- PyTorch + CUDA mismatch: install the correct torch wheel for your CUDA runtime. See https://pytorch.org/get-started/locally.
- If Streamlit reports "Load checkpoint models first (sidebar)", ensure you clicked the "Load Checkpoint Models" button after placing/uploading `progan_generator_final.pt` (or other checkpoint files).
- If you see TensorFlow/oneDNN messages and want to suppress them: set `TF_ENABLE_ONEDNN_OPTS=0` (see above).

## Notes
- The app provides heuristic fallbacks when a checkpoint is missing; results may be lower quality.
- Stable Diffusion inference on CPU is very slow — use GPU if possible.
- Keep ethical and legal considerations in mind when generating or analyzing images.