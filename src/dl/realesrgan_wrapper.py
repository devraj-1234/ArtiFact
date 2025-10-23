"""
Real-ESRGAN Wrapper for Art Restoration
- Auto-downloads weights on first use via Hugging Face Hub (ai-forever -> xinntao fallback)
- Simple single/batch restore APIs
"""

from __future__ import annotations

import cv2
import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

# Warn about NumPy 2.x (Real-ESRGAN stack prefers NumPy < 2)
try:
    from packaging.version import Version
    if Version(np.__version__) >= Version("2.0.0"):
        warnings.warn(
            f"Detected NumPy {np.__version__}. For best compatibility with Real-ESRGAN, "
            "use numpy==1.26.4 in this venv."
        )
except Exception:
    pass

# Try to import realesrgan/basicsr from pip
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    warnings.warn("Real-ESRGAN not installed. Install with: pip install realesrgan basicsr")

HF_CANDIDATES = [
    # Prefer new org first
    ("ai-forever/Real-ESRGAN", "weights/{name}.pth"),
    ("ai-forever/Real-ESRGAN", "{name}.pth"),
    # Fallback to original org
    ("xinntao/Real-ESRGAN", "weights/{name}.pth"),
    ("xinntao/Real-ESRGAN", "{name}.pth"),
]


def _repo_root() -> Path:
    # <repo>/src/dl/realesrgan_wrapper.py -> parents[2] == <repo>
    return Path(__file__).resolve().parents[2]


def ensure_realesrgan_weights(model_name: str, dst_dir: str | Path) -> str:
    """
    Download Real-ESRGAN weights via huggingface_hub, preferring ai-forever.
    Respects HTTP(S)_PROXY and HF token if set in env.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is required. Install with: pip install huggingface-hub") from e

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / f"{model_name}.pth"

    if dst_path.exists() and dst_path.stat().st_size > 0:
        return str(dst_path)

    last_err: Optional[Exception] = None
    for repo, pattern in HF_CANDIDATES:
        filename = pattern.format(name=model_name)
        try:
            local_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dst_dir),                # cache to our weights dir
                local_dir_use_symlinks=False           # write a real file, not a symlink
            )
            # Ensure final filename is consistent
            final_path = dst_path
            if Path(local_path) != final_path:
                Path(local_path).replace(final_path)
            if final_path.exists() and final_path.stat().st_size > 0:
                print(f"Downloaded {model_name} from {repo}/{filename} -> {final_path}")
                return str(final_path)
            last_err = RuntimeError("Downloaded file is empty")
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to download weights for {model_name}. "
        f"Tried repositories: {[c[0] for c in HF_CANDIDATES]} "
        f"Place the file manually at: {dst_path}\nLast error: {last_err}"
    )


class RealESRGANRestorer:
    """
    Wrapper for Real-ESRGAN pre-trained models.
    """

    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        device: str = "cpu",
        weights_dir: Optional[str] = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = False,
    ):
        if not REALESRGAN_AVAILABLE:
            raise ImportError("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")

        self.model_name = model_name
        self.device = device

        # Resolve and ensure weights exist (auto-download via HF)
        weights_dir = weights_dir or str(_repo_root() / "outputs" / "models" / "realesrgan")
        model_path = ensure_realesrgan_weights(model_name, weights_dir)

        # Choose architecture by model
        if model_name in ("RealESRGAN_x4plus", "RealESRNet_x4plus"):
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            scale = 4
        elif model_name == "RealESRGAN_x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            scale = 2
        elif model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            scale = 4
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            device=device,
        )

        print(f"Loaded Real-ESRGAN: {model_name} | scale={scale} | device={device}")

    def restore(
        self,
        image: np.ndarray,
        outscale: float = 1.0
        # face_enhance: bool = False,
    ) -> np.ndarray:
        """
        Enhance an image (BGR uint8).
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a non-empty numpy array (BGR, uint8)")
        output, _ = self.upsampler.enhance(img=image, outscale=outscale)
        return output

    def restore_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        outscale: float = 1.0
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Enhance an image from disk and optionally save.
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        enhanced = self.restore(img, outscale=outscale)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, enhanced)
        return enhanced, output_path

    def batch_restore(
        self,
        input_dir: str,
        output_dir: str,
        outscale: float = 1.0,
        extensions: Optional[List[str]] = None,
    ) -> None:
        """
        Batch enhance all images in a directory.
        """
        import glob

        exts = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(str(Path(input_dir) / f"*{ext}")))
        print(f"Found {len(files)} images")

        for i, p in enumerate(files, 1):
            name = Path(p).name
            out_p = str(Path(output_dir) / f"restored_{name}")
            try:
                self.restore_file(p, out_p, outscale=outscale)
                print(f"[{i}/{len(files)}] {name} -> {out_p}")
            except Exception as e:
                warnings.warn(f"Failed {name}: {e}")

        print(f"Batch complete. Output: {output_dir}")