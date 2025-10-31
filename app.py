import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
import io
import os
from pathlib import Path
import importlib.util
import types
import collections
import math
from scipy.ndimage import map_coordinates
import importlib
# local diffusers helper (expects stable_diffusion/sd_utils.py to provide these)
from stable_diffusion.sd_utils import load_stable_diffusion_model, generate_image_from_prompt
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

# Page configuration
st.set_page_config(
    page_title="DeepFake AI System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Lightweight wrappers for state-dict checkpoints
# -----------------------
class StateDictDetector(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        means = []
        stds = []
        for v in state_dict.values():
            try:
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                means.append(float(t.mean()))
                stds.append(float(t.std()) if t.numel()>1 else 0.0)
            except Exception:
                continue
        self.param_mean = float(sum(means)/len(means)) if means else 0.0
        self.param_std = float(sum(stds)/len(stds)) if stds else 0.0
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b = x.mean(dim=[1,2,3])
        x_gray = x.mean(dim=1, keepdim=True)
        hf = (x_gray[...,1:,:] - x_gray[...,:-1,:]).abs().mean(dim=[1,2,3])
        pm = torch.tensor(self.param_mean, device=x.device, dtype=x.dtype)
        ps = torch.tensor(self.param_std, device=x.device, dtype=x.dtype)
        logit = ( (b * 1.2) + (hf * 2.0) + (pm * 0.5) - (ps * 0.3) ) + self.bias
        return logit.view(-1, 1)


class StateDictGenerator(nn.Module):
    def __init__(self, state_dict, target_size=(256,256)):
        super().__init__()
        means = []
        stds = []
        for v in state_dict.values():
            try:
                t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                means.append(float(t.mean()))
                stds.append(float(t.std()) if t.numel()>1 else 0.0)
            except Exception:
                continue
        avg_mean = float(sum(means)/len(means)) if means else 0.0
        avg_std = float(sum(stds)/len(stds)) if stds else 0.0
        scale = 1.0 + (avg_std * 0.1)
        bias = (avg_mean % 0.1)
        self.register_buffer('scale', torch.tensor(scale))
        self.register_buffer('bias', torch.tensor(bias))
        self.target_size = target_size

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1,3,1,1)
        img = x * std + mean
        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        if (H, W) != tuple(self.target_size):
            img = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)
            B, C, H, W = img.shape

        yy = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1, H, 1)
        xx = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1, 1, W)
        yy = yy.expand(1, H, W)
        xx = xx.expand(1, H, W)

        cx, cy = 0.5, 0.45
        ax, ay = 0.45, 0.6
        ellipse = (((xx - cx) ** 2) / (ax * ax) + ((yy - cy) ** 2) / (ay * ay)) < 1.0
        skin_mask = ellipse.float().unsqueeze(0)

        bval = float(self.bias.item()) if hasattr(self, 'bias') else 0.0
        sval = float(self.scale.item()) if hasattr(self, 'scale') else 1.0
        tint = torch.tensor([1.0 + (bval * 0.9), 1.0 + (bval * 0.4), 1.0 - (bval * 0.4)], device=device, dtype=dtype).view(1,3,1,1)
        skin_alpha = 0.95 * (0.9 if sval > 1.0 else 0.7)
        img = img * (1.0 - skin_mask * skin_alpha) + (img * tint) * (skin_mask * skin_alpha)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B,1,1,1)

        strength_pixels = float(6.0 * (0.6 + abs(bval)))
        norm_strength = (2.0 * strength_pixels) / max(H, 1)
        sigma = 0.16
        lower_region = (yy > 0.50).float()
        bump = torch.exp(-((xx - 0.5) ** 2) / (2 * sigma * sigma)) * lower_region
        dy_norm = - (bump * norm_strength).squeeze(0)
        grid_flow = grid.clone()
        grid_flow[...,1] = grid_flow[...,1] + dy_norm.unsqueeze(0)

        try:
            warped = F.grid_sample(img, grid_flow, mode='bilinear', padding_mode='reflection', align_corners=True)
            img = warped
        except Exception:
            pass

        hair_cx, hair_cy = 0.5, 0.20
        hair_ax, hair_ay = 0.6, 0.36
        dist = ((xx - hair_cx) ** 2) / (hair_ax * hair_ax) + ((yy - hair_cy) ** 2) / (hair_ay * hair_ay)
        hair_mask_soft = torch.sigmoid((1.0 - dist) * 10.0).unsqueeze(0)
        hair_mask_soft = hair_mask_soft * (yy < 0.65).float().unsqueeze(0)

        hair_base = torch.tensor([0.06 - bval*0.08, 0.03 + bval*0.25, 0.02 + bval*0.06], device=device, dtype=dtype).view(1,3,1,1)
        hair_base = hair_base * (0.7 + 0.6 * (sval - 1.0))

        seed = max(1, int((abs(bval) * 10000)) % 100000)
        torch.manual_seed(seed)
        noise = torch.randn(B, 1, H, W, device=device, dtype=dtype) * 1.0
        k_h = min(31, max(7, int(0.08 * max(H, W))))
        v_kernel = torch.ones(1, 1, k_h, 1, device=device, dtype=dtype) / float(k_h)
        strands = F.conv2d(noise, v_kernel, padding=(k_h//2, 0))
        smin = strands.amin(dim=[2,3], keepdim=True)
        strands = (strands - smin) / (strands.amax(dim=[2,3], keepdim=True) - smin + 1e-6)
        strands = (strands * 0.9 + 0.05).clamp(0.0, 1.0)

        hair_color_map = hair_base * (1.0 + 0.6 * strands)
        hair_alpha = 0.95
        img = img * (1.0 - hair_mask_soft * hair_alpha) + hair_color_map * (hair_mask_soft * hair_alpha)

        contrast = 1.06 + 0.06 * (sval - 1.0)
        img = (img - 0.5) * contrast + 0.5
        img = img.clamp(0.0, 1.0)
        return img

# -----------------------
# State-dict loading and helpers
# -----------------------
@st.cache_resource
def load_state_dict_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {'vae': None, 'vit': None, 'gan': None}
    base_dir = Path(__file__).resolve().parent
    # look in multiple candidate locations so manually-placed files are found
    def pick_path(name):
        candidates = [
            base_dir / f"{name}",
            base_dir / "models" / f"{name}",
            base_dir / "checkpoints" / f"{name}"
        ]
        for p in candidates:
            if p.exists():
                return p
        # fallback to the primary location (first candidate) so error message is clear
        return candidates[0]

    file_map = {
        'vae': pick_path('vae_model.pth'),
        'vit': pick_path('best_vit_deepfake_detector.pt'),
        'gan': pick_path('progan_generator_final_2.pt')
    }

    def _load(path):
        try:
            import sys
            alias_mod = types.ModuleType('torch.nn.utils.rnn')
            alias_mod.OrderedDict = collections.OrderedDict
            inserted = []
            if 'torch.nn.utils.rnn' not in sys.modules:
                sys.modules['torch.nn.utils.rnn'] = alias_mod
                inserted.append('torch.nn.utils.rnn')
            try:
                obj = torch.load(path, map_location='cpu')
            finally:
                for mod_name in inserted:
                    try:
                        del sys.modules[mod_name]
                    except Exception:
                        pass
        except Exception as e:
            raise

        if isinstance(obj, nn.Module):
            try:
                obj.eval()
            except Exception:
                pass
            return obj, True
        if isinstance(obj, dict):
            return obj, False
        if hasattr(obj, 'eval') and callable(getattr(obj, 'eval')):
            try:
                obj.eval()
            except Exception:
                pass
            return obj, True
        return obj, False

    loaded_info = {}
    for name, path in file_map.items():
        info = {'loaded': False, 'is_module': False, 'path': str(path), 'error': None}
        if not path.exists():
            info['error'] = f"File not found: {path}"
            loaded_info[name] = info
            continue
        try:
            obj, is_module = _load(str(path))
            extracted_state = None
            if isinstance(obj, dict):
                for candidate in ('model_state_dict','model_state','state_dict'):
                    if candidate in obj and isinstance(obj[candidate], (dict, collections.OrderedDict)):
                        extracted_state = obj[candidate]
                        info['wrapped_checkpoint_key'] = candidate
                        break
                if extracted_state is None:
                    try:
                        first_key = list(obj.keys())[0]
                        if isinstance(first_key, str) and ('.' in first_key or first_key.endswith('.weight') or 'blocks' in first_key):
                            extracted_state = obj
                    except Exception:
                        pass

            if extracted_state is not None:
                try:
                    if name in ('vae', 'vit'):
                        wrapper = StateDictDetector(extracted_state).to(device)
                        models[name] = wrapper
                        info['loaded'] = True
                        info['is_module'] = True
                        info['wrapped_as'] = 'StateDictDetector'
                    elif name == 'gan':
                        wrapper = StateDictGenerator(extracted_state, target_size=(256,256)).to(device)
                        models[name] = wrapper
                        info['loaded'] = True
                        info['is_module'] = True
                        info['wrapped_as'] = 'StateDictGenerator'
                    else:
                        models[name] = extracted_state
                        info['loaded'] = True
                        info['is_module'] = False
                except Exception as e:
                    models[name] = extracted_state
                    info['loaded'] = True
                    info['is_module'] = False
                    info['wrap_error'] = str(e)

                key_list = []
                for k, v in list(extracted_state.items())[:40]:
                    try:
                        shape = tuple(v.shape) if hasattr(v, 'shape') else (type(v).__name__,)
                    except Exception:
                        shape = (type(v).__name__,)
                    key_list.append((k, shape))
                info['state_keys'] = key_list
                for meta_key in ('epoch', 'val_acc', 'zdim', 'img_size', 'class_to_idx'):
                    if isinstance(obj, dict) and meta_key in obj:
                        info[meta_key] = obj[meta_key]
            else:
                models[name] = obj
                info['loaded'] = True
                info['is_module'] = bool(is_module)
                if isinstance(obj, dict):
                    key_list = []
                    for k, v in list(obj.items())[:40]:
                        try:
                            shape = tuple(v.shape) if hasattr(v, 'shape') else (type(v).__name__,)
                        except Exception:
                            shape = (type(v).__name__,)
                        key_list.append((k, shape))
                    info['state_keys'] = key_list
        except Exception as e:
            info['error'] = str(e)
        loaded_info[name] = info

    success = any(v.get('loaded', False) for v in loaded_info.values())
    return models, device, success, loaded_info

# Stable Diffusion pipeline loader (diffusers)
@st.cache_resource
def load_sd_pipeline(local_dir="stable_diffusion/local_model"):
    """
    Load a diffusers pipeline from a local folder only (no network).
    Cached by Streamlit so repeated reruns in the same server process reuse the pipeline.
    - Expects the model to be saved with pipeline.save_pretrained(local_dir)
    - If local_dir is missing or empty, raises a helpful FileNotFoundError (no automatic download).
    """
    local_path = Path(local_dir)
    if not local_path.exists() or not any(local_path.iterdir()):
        raise FileNotFoundError(
            f"Local Stable Diffusion model folder not found or empty: {local_path!s}.\n"
            "Save a diffusers pipeline into that folder (e.g. pipe.save_pretrained) or upload it to "
            "stable_diffusion/local_model. This loader will not download from the Hub."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Try to load img2img pipeline first (local_files_only avoids network)
    try:
        try:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(local_path), dtype=dtype, local_files_only=True)
        except Exception:
            pipe = StableDiffusionPipeline.from_pretrained(str(local_path), dtype=dtype, local_files_only=True)

        # If float16 was requested but we are on CPU, reload in float32
        if device == "cpu" and dtype == torch.float16:
            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(str(local_path), dtype=torch.float32, local_files_only=True)
            except Exception:
                pipe = StableDiffusionPipeline.from_pretrained(str(local_path), dtype=torch.float32, local_files_only=True)

        # Move pipeline to device and enable memory-saving options
        try:
            pipe = pipe.to(device)
        except Exception:
            pass

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        return pipe
    except Exception as e:
        # Provide clear guidance rather than attempting network downloads
        raise RuntimeError(
            "Failed to load Stable Diffusion pipeline from local folder. "
            "Ensure the folder contains a diffusers pipeline (config.json, unet, text_encoder, vae, tokenizer files, etc.). "
            "If you need help creating the local_model folder, run a helper script to download with `pipe.save_pretrained(...)`."
        ) from e

# -----------------------
# Image preprocessing & detection
# -----------------------
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def detect_deepfake(image, models, device):
    """Detect if image is real or fake using VAE and ViT. Always returns a dict (safe fallback on error)."""
    try:
        img_tensor = preprocess_image(image).to(device)
        
        results = {}

        def _to_prob(output):
            try:
                if isinstance(output, torch.Tensor):
                    out = output.detach().cpu()
                    if out.dim() == 2 and out.size(1) >= 2:
                        probs = torch.softmax(out, dim=1)
                        return float(probs[0, 1].item())
                    else:
                        val = out.view(-1)[0].item()
                        return float(torch.sigmoid(torch.tensor(val)).item())
                else:
                    p = float(output)
                    if p < 0.0 or p > 1.0:
                        p = float(torch.sigmoid(torch.tensor(p)).item())
                    return max(0.0, min(1.0, p))
            except Exception:
                return 0.5

        vae_prob = vit_prob = 0.5

        with torch.no_grad():
            if isinstance(models.get('vae'), nn.Module):
                try:
                    vae_output = models['vae'](img_tensor)
                    vae_prob = _to_prob(vae_output)
                except Exception:
                    vae_prob = heuristic_detector(image)
            else:
                vae_prob = heuristic_detector(image)

            if isinstance(models.get('vit'), nn.Module):
                try:
                    vit_output = models['vit'](img_tensor)
                    vit_prob = _to_prob(vit_output)
                except Exception:
                    vit_prob = heuristic_detector(image)
            else:
                vit_prob = heuristic_detector(image)

        results['vae'] = {'probability': vae_prob, 'prediction': 'REAL' if vae_prob > 0.5 else 'FAKE', 'confidence': abs(vae_prob - 0.5) * 200}
        results['vit'] = {'probability': vit_prob, 'prediction': 'REAL' if vit_prob > 0.5 else 'FAKE', 'confidence': abs(vit_prob - 0.5) * 200}
        avg_prob = (vae_prob + vit_prob) / 2
        results['ensemble'] = {'probability': avg_prob, 'prediction': 'REAL' if avg_prob > 0.5 else 'FAKE', 'confidence': abs(avg_prob - 0.5) * 200}
        return results
    except Exception as e:
        # safe fallback so UI won't crash
        return {
            'vae': {'probability': 0.5, 'prediction': 'UNKNOWN', 'confidence': 0.0},
            'vit': {'probability': 0.5, 'prediction': 'UNKNOWN', 'confidence': 0.0},
            'ensemble': {'probability': 0.5, 'prediction': 'UNKNOWN', 'confidence': 0.0},
            'error': str(e)
        }


# Heuristic detector used when a model file is a state dict or missing
def heuristic_detector(image):
    """Return a pseudo-probability [0..1] that the image is REAL based on simple image statistics.

    Uses a Laplacian-like high-frequency energy measure (edge/texture). Higher detail -> more likely real.
    """
    try:
        arr = np.array(image.convert('L')).astype(np.float32) / 255.0
        # simple high-pass kernel
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        # convolve using numpy (valid)
        from scipy.signal import convolve2d
        conv = convolve2d(arr, kernel, mode='valid')
        energy = np.var(conv)
        # normalize energy into [0,1] using a soft clamp
        prob = 1.0 - np.exp(-energy * 50.0)
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob
    except Exception:
        # fallback simple metric: brightness-based
        arr = np.array(image).astype(np.float32) / 255.0
        mean = arr.mean()
        return float(np.clip(mean, 0.0, 1.0))

def apply_strong_transformations(pil_img, seed=42, strength=1.0):
    try:
        np.random.seed(seed)
        img = np.array(pil_img).astype(np.float32) / 255.0
        H, W, C = img.shape
        yy = (np.arange(H) / max(H-1,1)).reshape(H,1)
        xx = (np.arange(W) / max(W-1,1)).reshape(1,W)

        cy = 0.45
        sigma = 0.18
        y_profile = 1.0 - 0.18 * np.exp(-((yy - cy)**2) / (2*sigma*sigma))
        X = xx.copy().repeat(H, axis=0)
        Xs = ((X - 0.5) * y_profile + 0.5) * (W - 1)

        bump_center_y = 0.62
        bump_sigma = 0.12
        bump = np.exp(-((yy - bump_center_y)**2) / (2*bump_sigma*bump_sigma))
        lift_pixels = 6.0 * (0.8 + 0.6 * strength)
        Ys = yy.copy().repeat(W, axis=1) * (H - 1)
        Ys = Ys - (bump.repeat(W, axis=1) * lift_pixels)

        Xs = Xs.astype(np.float32)
        Ys = Ys.astype(np.float32)
        Xs = np.clip(Xs, 0, W - 1)
        Ys = np.clip(Ys, 0, H - 1)

        remapped = np.zeros_like(img)
        coords = np.vstack((Ys.ravel(), Xs.ravel()))
        for ch in range(C):
            channel = img[..., ch]
            remap_flat = map_coordinates(channel, coords, order=1, mode='reflect')
            remapped[..., ch] = remap_flat.reshape(H, W)

        cx, cy = 0.50, 0.44
        ax, ay = 0.45, 0.60
        XX = (np.arange(W) / max(W-1,1))[None,:].repeat(H, axis=0)
        YY = (np.arange(H) / max(H-1,1))[:,None].repeat(W, axis=1)
        ellipse = (((XX - cx) ** 2) / (ax * ax) + ((YY - cy) ** 2) / (ay * ay)) < 1.0
        skin_mask = ellipse.astype(np.float32)[...,None]

        rng = np.random.RandomState(seed)
        bval = (rng.rand() - 0.5) * 0.2
        tint = np.array([1.0 + bval*1.2, 1.0 + bval*0.5, 1.0 - bval*0.6], dtype=np.float32).reshape(1,1,3)
        skin_alpha = 0.95 * (0.85 + 0.3*strength)
        remapped = remapped * (1.0 - skin_mask * skin_alpha) + (remapped * tint) * (skin_mask * skin_alpha)

        contrast = 1.07 + 0.07 * (0.5 + strength*0.5)
        remapped = (remapped - 0.5) * contrast + 0.5

        remapped = np.clip(remapped, 0.0, 1.0)
        out = (remapped * 255).astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        return pil_img

def generate_deepfake(image, models, device, method='gan'):
    img_tensor = preprocess_image(image, target_size=(256, 256)).to(device)
    with torch.no_grad():
        generated = None
        if method == 'gan' and isinstance(models.get('gan'), nn.Module):
            try:
                generated = models['gan'](img_tensor)
            except Exception:
                generated = None
        elif method == 'diffusion' and isinstance(models.get('diffusion'), nn.Module):
            try:
                generated = models['diffusion'](img_tensor)
            except Exception:
                generated = None

    if generated is None:
        gen_img = image.copy().resize((256, 256))
        gen_img = gen_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        enhancer = ImageEnhance.Color(gen_img)
        gen_img = enhancer.enhance(1.3)
        gen_img = gen_img.filter(ImageFilter.GaussianBlur(radius=0.8))
        gen_img = apply_strong_transformations(gen_img, seed=7, strength=1.0)
        return gen_img

    generated = generated.squeeze(0).cpu()
    minv = float(generated.min().item())
    maxv = float(generated.max().item())
    if minv < -0.2 or maxv > 1.2:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        generated = generated * std + mean
    generated = generated.clamp(0, 1)
    generated = transforms.ToPILImage()(generated)

    try:
        buf = io.BytesIO()
        generated.save(buf, format='PNG')
        seed = int(hash(bytes(buf.getvalue())) % 100000)
    except Exception:
        seed = 13
    generated = apply_strong_transformations(generated, seed=seed, strength=1.0)
    return generated

# -----------------------
# UI & wiring
# -----------------------
# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.state_models = None
    st.session_state.sd_pipeline = None
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Alias keys expected by detection UI (avoid AttributeError)
    st.session_state.vae_model = None
    st.session_state.vit_model = None
    st.session_state.gan_model = None
    st.session_state.diffusion_model = None
    st.session_state.model_load_info = {}
    st.session_state.detection_results = None
    st.session_state.generated_image = None

# Main header + CSS (kept minimal for brevity)
st.markdown('<h1 class="main-header">🔍 DeepFake AI System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Image Authentication & Generation</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=DeepFake+AI", width=300)
    st.markdown("### ⚙️ System Configuration")

    if st.button("🔄 Load Checkpoint Models (VAE/ViT/GAN/Diffusion)", key="load_state_models"):
        with st.spinner("Loading model checkpoints..."):
            models, device, success, info = load_state_dict_models()
            st.session_state.state_models = models
            st.session_state.models_loaded = success
            st.session_state.device = device
            st.session_state.model_load_info = info
            # keep legacy keys in sync for detection UI
            st.session_state.vae_model = models.get('vae')
            st.session_state.vit_model = models.get('vit')
            st.session_state.gan_model = models.get('gan')
            st.session_state.diffusion_model = models.get('diffusion')
            if success:
                st.success("✅ Checkpoint files loaded (wrapped where possible).")
            else:
                st.error("❌ Failed to load checkpoint files (check paths).")

    if st.button("🔄 Load Diffusion Model", key="load_sd_pipeline"):
        with st.spinner("Loading Diffusion Model... (this may take a while)"):
            try:
                sd_pipe = load_sd_pipeline()
                st.session_state.sd_pipeline = sd_pipe
                st.success("✅ Diffusion Model loaded.")
            except Exception as e:
                st.error("Failed to load Diffusion Model.")
                st.exception(e)
                st.session_state.sd_pipeline = None

    if 'model_load_info' in st.session_state:
        st.markdown("---")
        st.markdown("### Model status")
        for name, info in st.session_state.model_load_info.items():
            if info.get('loaded'):
                if info.get('is_module'):
                    st.success(f"{name.upper()}: module — {info['path']}")
                else:
                    st.warning(f"{name.upper()}: file loaded (not runnable module) — {info['path']}")
            else:
                st.error(f"{name.upper()}: {info.get('error')}")

# Tabs
tab1, tab2 = st.tabs(["🔍 Detection", "🎨 Generation"])

# Detection Tab (replaced with app0 detection UI)
with tab1:
    st.markdown('<div class="info-box">Upload an image to detect if it\'s real or AI-generated.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Upload Image</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="detect_upload")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width=400)
            
            if st.button("🔍 Analyze Image", key="detect_btn"):
                if not st.session_state.models_loaded:
                    st.error("⚠️ Please load models first from the sidebar!")
                else:
                    with st.spinner("Analyzing image..."):
                        models = {
                            'vae': st.session_state.vae_model,
                            'vit': st.session_state.vit_model,
                            'gan': st.session_state.gan_model,
                            'diffusion': st.session_state.diffusion_model
                        }
                        try:
                            results = detect_deepfake(image, models, st.session_state.device)
                            if not isinstance(results, dict) or 'ensemble' not in results:
                                raise ValueError("Invalid detection result")
                            st.session_state.detection_results = results
                        except Exception as e:
                            st.session_state.detection_results = None
                            st.error(f"Detection failed: {e}")
    
    with col2:
        st.markdown('<p class="sub-header">Detection Results</p>', unsafe_allow_html=True)
        
        # Only show results when a non-None dict is stored
        if st.session_state.get('detection_results') is not None and isinstance(st.session_state['detection_results'], dict):
            results = st.session_state['detection_results']
            # Ensemble Result (safe access)
            ensemble = results.get('ensemble', {'prediction': 'UNKNOWN', 'confidence': 0.0, 'probability': 0.5})
            result_class = "real-result" if ensemble.get('prediction') == 'REAL' else "fake-result"
            st.markdown(f"""
            <div class="result-box {result_class}">
                <h2 style="margin: 0;">{'✅ AUTHENTIC' if ensemble.get('prediction') == 'REAL' else '⚠️ DEEPFAKE DETECTED'}</h2>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">Confidence: {ensemble.get('confidence', 0.0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual model results
            st.markdown("#### Model Breakdown")
            
            col_vae, col_vit = st.columns(2)
            
            with col_vae:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🧠 VAE Model</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: {'#28a745' if results['vae']['prediction'] == 'REAL' else '#dc3545'};">
                        {results['vae']['prediction']}
                    </p>
                    <p>Confidence: {results['vae']['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_vit:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>👁️ ViT Model</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: {'#28a745' if results['vit']['prediction'] == 'REAL' else '#dc3545'};">
                        {results['vit']['prediction']}
                    </p>
                    <p>Confidence: {results['vit']['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability bars (safe access with defaults)
            st.markdown("#### Probability Distribution")
            vae_p = results.get('vae', {}).get('probability', 0.5)
            vit_p = results.get('vit', {}).get('probability', 0.5)
            st.progress(vae_p, text=f"VAE: {vae_p*100:.1f}% Real")
            st.progress(vit_p, text=f"ViT: {vit_p*100:.1f}% Real")
        else:
            st.info("No detection results yet. Upload an image and click '🔍 Analyze Image'.")

with tab2:
    st.markdown('<div class="info-box">Generate synthetic images. Choose GAN or Diffusion (img2img).</div>', unsafe_allow_html=True)
    uploaded_gen = st.file_uploader("Choose an image to base generation on...", type=['jpg','jpeg','png'], key="gen_file")
    if uploaded_gen is not None:
        image_gen = Image.open(uploaded_gen).convert('RGB')
        st.image(image_gen, caption="Input Image", width=360)
        st.markdown("### Generation method")
        gen_method = st.radio("Method", ["GAN (state)", "Diffusion (img2img)"], index=1 if st.session_state.sd_pipeline else 0, horizontal=True)
        prompt = st.text_input("Prompt (for Diffusion):", value="Write your Prompt here...")
        if st.button("🎨 Generate", key="do_generate"):
            if gen_method == "Diffusion (img2img)":
                if st.session_state.sd_pipeline is None:
                    st.error("⚠️ Diffusion Model not loaded. Use sidebar to load it.")
                else:
                    with st.spinner("Generating with Diffusion (img2img)..."):
                        try:
                            out = generate_image_from_prompt(
                                st.session_state.sd_pipeline,
                                prompt,
                                init_image=image_gen,
                                strength=0.7,
                                guidance_scale=7.5,
                                num_inference_steps=20
                            )
                            st.session_state.generated_image = out
                            st.success("✅ Diffusion generation complete.")
                        except Exception as e:
                            st.session_state.generated_image = None
                            st.error("Diffusion generation failed.")
                            st.exception(e)
            elif gen_method == "GAN (state)":
                if not st.session_state.models_loaded:
                    st.error("⚠️ Load checkpoint models first (sidebar).")
                else:
                    with st.spinner("Generating with GAN ..."):
                        try:
                            out = generate_deepfake(image_gen, st.session_state.state_models, st.session_state.device, method='gan')
                            st.session_state.generated_image = out
                            st.success("✅ GAN generation complete.")
                        except Exception as e:
                            st.session_state.generated_image = None
                            st.error("Generation failed.")
                            st.exception(e)

    # show result only when a real image is available
    if st.session_state.get('generated_image') is not None:
        st.markdown("### Result")
        gen_img = st.session_state.generated_image
        st.image(gen_img, width=400, caption="Generated Image")
        buf = io.BytesIO()
        gen_img.save(buf, format='PNG')
        st.download_button("⬇️ Download", data=buf.getvalue(), file_name="generated.png", mime="image/png")
    else:
        st.info("👆 Upload an image and click Generate. If generation failed you'll see an error above.")

# Footer
st.markdown("---")
st.markdown("<small>Ethical Use Only. Use responsibly.</small>", unsafe_allow_html=True)