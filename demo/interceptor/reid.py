"""Person ReID embedder. OSNet x0_25 + MSMT17 -> 512-D L2-normalized embedding per bbox crop.
Same person -> small cosine distance across pose/lighting. Enables identity persistence + re-acquisition.

Public:
    ReidEmbedder(device).embed_batch(crops_bgr) -> Tensor (N, 512)
    cosine_distance(a, b) -> float in [0, 2]
    resolve_weights_path(path) -> str
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import torchreid
from torchreid.reid.utils.torchtools import load_pretrained_weights

from interceptor.constants import (
    LOCK_EMBEDDING_EMA_ALPHA,
    REID_INPUT_SIZE_HW,
    REID_MODEL_NAME,
    REID_PIXEL_MEAN,
    REID_PIXEL_STD,
    REID_WEIGHTS_PATH,
)


def resolve_weights_path(path: str = REID_WEIGHTS_PATH) -> str:
    """Return absolute weight-file path. Raises FileNotFoundError if missing. Weights vendored in models/."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"ReID weights missing at {abs_path}. "
            f"Restore from git (file is vendored at models/osnet_x0_25_msmt17.pt)."
        )
    return abs_path



class ReidEmbedder:
    """OSNet person embedder. Handles BGR→RGB, resize, normalize. Callers pass raw uint8 BGR crops."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = resolve_weights_path()

        model = torchreid.models.build_model(
            name=REID_MODEL_NAME,
            num_classes=1000,        # ignored at inference
            pretrained=False,
        )
        load_pretrained_weights(model, weight_path)
        model.eval().to(self.device)
        self.model = model

        # Norm tensors pre-built on device — avoids host→device copy each embed_batch.
        self._mean = torch.tensor(REID_PIXEL_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std  = torch.tensor(REID_PIXEL_STD,  device=self.device).view(1, 3, 1, 1)
        self._target_h, self._target_w = REID_INPUT_SIZE_HW

        print(f"[INFO] ReidEmbedder ready on {self.device} (model={REID_MODEL_NAME})")

    @torch.no_grad()
    def embed_batch(self, crops_bgr: Iterable[np.ndarray]) -> Optional[torch.Tensor]:
        """Embed N bbox crops in one forward pass. Returns (N, 512) L2-normalized on self.device, or None if empty."""
        crops = [c for c in crops_bgr if c is not None and c.size > 0]
        if not crops:
            return None

        # Resize each crop before stacking — they may differ in shape. INTER_LINEAR matches torchreid default.
        resized = [
            cv2.resize(c, (self._target_w, self._target_h), interpolation=cv2.INTER_LINEAR)
            for c in crops
        ]

        # NHWC uint8 -> BGR→RGB -> NCHW -> /255 -> normalize.
        batch_np = np.stack(resized, axis=0)[..., ::-1]
        # .copy() — ::-1 view must be contiguous before from_numpy on Windows.
        batch = torch.from_numpy(batch_np.copy()).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = (batch - self._mean) / self._std

        embeddings = self.model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def set_device(self, device: str) -> None:
        """Move model + norm tensors to cuda/cpu."""
        self.model.to(device)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.device = device


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance between L2-normalized embeddings. Inputs 1-D (512,) or (1, 512).
    Returns float in [0, 2]: 0=same dir, 1=orthogonal, 2=opposite.
    """
    a = a.flatten()
    b = b.flatten()
    return float(1.0 - torch.dot(a, b).item())


def update_embedding_ema(old: torch.Tensor, new: torch.Tensor,
                         alpha: float = LOCK_EMBEDDING_EMA_ALPHA) -> torch.Tensor:
    """EMA-blend locked embedding toward fresh match. Re-normalizes."""
    mixed = (1.0 - alpha) * old + alpha * new
    return F.normalize(mixed, p=2, dim=0)


def crop_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """Return frame[y1:y2, x1:x2] clamped to bounds, or None if degenerate."""
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
