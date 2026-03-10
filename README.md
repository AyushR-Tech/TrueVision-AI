# DeepFake AI System

> **Important:** this repository does **not** include large model weights (Stable Diffusion, GAN, etc.). For deployment use or collaboration, the app either downloads models at runtime or you must place them manually. Keeping weights out of git keeps the repo light and works with Streamlit Cloud.

A Streamlit app that provides:
- Deepfake detection (VAE + ViT + heuristic fallback)
- Image generation (GAN via a checkpoint wrapper + Stable Diffusion img2img via a local diffusers pipeline)
- Sidebar helpers to upload and load checkpoints and the local Stable Diffusion pipeline
- Lightweight ethics/usage scaffolding and basic UI

This README was updated to reflect recent changes: the app now includes a GAN upload control, checkpoint loader that no longer attempts to load a legacy diffusion state file, a cached local-only Stable Diffusion loader, and integrated detection UI.

---

## File layout (important)
- app.py — main Streamlit application (UI + logic)
- stable_diffusion/sd_utils.py — helper to load a local diffusers pipeline and run img2img
- scripts/download_diffusers_model.py — (optional) helper to download & save a diffusers pipeline locally
- progan_generator_final.pt — GAN checkpoint (place in project root or upload via the app sidebar)
- vae_model.pth, best_vit_deepfake_detector.pt — detector checkpoints (optional)
- stable_diffusion/local_model/ — local diffusers pipeline directory. **Not required**; if missing the app will attempt to download a model from Hugging Face. Do **not** commit this directory to git (it contains multiple gigabytes of weights).
- requirements.txt — Python dependencies
- .gitignore, .gitattributes — repository config (do NOT commit large model binaries)

---

## Key changes / notes
- The checkpoint loader now looks for VAE, ViT and GAN checkpoints only (no state-dict diffusion file).
- The app provides a GAN checkpoint uploader in the sidebar; uploaded checkpoint is saved into the project folder so you can press "Load Checkpoint Models".
- Stable Diffusion is loaded preferably from a local folder (`stable_diffusion/local_model`) but the loader now has a fallback: if the folder is missing or empty the app will fetch `runwayml/stable-diffusion-v1-5` from Hugging Face on first run. This makes the repo much lighter and avoids deployment crashes.
- SciPy is required by default (used for remapping and convolution). The app offers a lower-quality fallback if SciPy is unavailable, but install SciPy for best results.

---

## Prerequisites
- Python 3.8–3.11
- (Recommended) Virtual environment: venv or conda
- For GPU: install a matching PyTorch wheel for your CUDA version (see https://pytorch.org/get-started/locally)

---

## Install & setup (Windows PowerShell)
1. Create & activate venv
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Install PyTorch (pick correct command from PyTorch website). Examples:
   - CPU-only:
     ```powershell
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
     ```
   - CUDA (replace cu118 or version as appropriate):
     ```powershell
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

3. Install the rest of dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. (Optional) Reduce TensorFlow oneDNN info messages:
   ```powershell
   $env:TF_ENABLE_ONEDNN_OPTS = "0"
   ```

---

## Where to place model files (important)
- GAN checkpoint (`progan_generator_final.pt`)
  - Place in project root (same folder as `app.py`) OR in `models/`.  
  - Alternatively upload via the app sidebar — the app will save it as `progan_generator_final.pt` to the project folder.
- Detector checkpoints (`vae_model.pth`, `best_vit_deepfake_detector.pt`)
  - Place in project root or `models/`.
- Stable Diffusion (img2img)
  - Must be a diffusers pipeline saved locally at `stable_diffusion/local_model/`. The folder must include pipeline files (config.json, unet, vae, text_encoder/tokenizer, etc.).
  - You can create this local folder by running the included helper script or by saving a pipeline in another environment and zipping / copying it into this folder.

Helper to download & save a local SD pipeline (example):
- Edit `scripts/download_diffusers_model.py` with your HF token and desired model id, then run:
  ```powershell
  $env:HF_TOKEN="your_hf_token"
  python .\scripts\download_diffusers_model.py
  ```

---

## Run the app
From project root:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
Open the URL printed by Streamlit (usually http://localhost:8501).

---

## Typical usage flow (in the app)
1. Sidebar:
   - Upload your `progan_generator_final.pt` with "Upload GAN checkpoint" (optional).
   - Click "🔄 Load Checkpoint Models" to load VAE / ViT / GAN wrappers (this enables detection and GAN generation).
   - Click "🔄 Load Stable Diffusion Pipeline" to load the local SD pipeline (one-time per session); the pipeline is cached.
2. Detection tab:
   - Upload an image and click "🔍 Analyze Image" to get VAE / ViT / ensemble predictions and the heuristic fallback.
3. Generation tab:
   - Upload an image, choose "GAN (state)" to use the GAN checkpoint, or "Diffusion (img2img)" to use the loaded local SD pipeline, then click "🎨 Generate".

---

## Sharing models with collaborators
- Do NOT commit large model binaries into git (GitHub blocks >100MB). Use one of:
  - Git LFS (track `.pt`, `.pth`, `.ckpt`, `.bin`) — recommended for moderately large files.
  - Upload zipped `stable_diffusion/local_model` and GAN checkpoint to Google Drive / Dropbox / GitHub Releases; collaborators download and extract into project folders.
- Example: zip `stable_diffusion/local_model` and instruct collaborators to extract into `stable_diffusion/local_model/`.

---

## Troubleshooting / common issues
- ModuleNotFoundError: SciPy — install via `pip install scipy`.
- "Load checkpoint models first" — ensure you pressed the "Load Checkpoint Models" button in the sidebar after placing/uploading `progan_generator_final.pt`.
- Stable Diffusion load shows "Loading pipeline components..." — this is normal during initialization; if you did not use a local model folder the loader may try to download or fail. Ensure `stable_diffusion/local_model` exists and contains the saved pipeline.
- PyTorch + CUDA mismatch — install the correct torch wheel for your CUDA runtime.
- If you intend to track model files in git, enable Git LFS:
  ```powershell
  git lfs install
  git lfs track "*.pt" "*.pth" "*.ckpt" "*.bin"
  git add .gitattributes
  git commit -m "Track model files with Git LFS"
  ```

---

## Security & ethics
- The app includes an Ethical Safeguards scaffold but is intended for research / educational use only.
- Respect privacy, copyright and legal restrictions when using datasets or deploying generated images.

---

## Additional notes
- The app provides heuristic fallbacks when checkpoints are missing; results may be lower quality.
- Stable Diffusion on CPU is slow — use GPU where available.
- If you want a direct zipped package of `stable_diffusion/local_model` for distribution, upload it to a cloud host and provide the download link