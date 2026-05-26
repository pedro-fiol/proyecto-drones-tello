"""
interceptor/reid.py

Person re-identification (ReID) embedder. Wraps an OSNet x0_25 model trained
on MSMT17 (torchreid model zoo) and turns each person bbox crop into a 512-D
L2-normalized embedding. Two embeddings of the same person have small cosine
distance even across pose/lighting changes — enables identity persistence
across frames and out-of-frame re-acquisition.

Public surface:
    ReidEmbedder(device).embed_batch(crops_bgr) -> Tensor (N, 512)
    cosine_distance(a, b) -> float in [0, 2]
    resolve_weights_path(path) -> str

No drone SDK calls, no drawing, no FSM logic. Pure perception adjunct.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Deep import is fine — torchreid's eager root __init__ has heavy deps, but we
# already paid that cost during package install (tensorboard).
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


# ---------------------------------------------------------------------------
# Weights helper
# ---------------------------------------------------------------------------

def resolve_weights_path(path: str = REID_WEIGHTS_PATH) -> str:
    """Return absolute weight-file path. Raises FileNotFoundError if missing.

    Weights are vendored in models/ (see .gitignore exception). If the file
    is absent the operator deleted/moved it — fix locally rather than auto-
    fetching from a flaky Drive link.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(
            f"ReID weights missing at {abs_path}. "
            f"Restore from git (file is vendored at models/osnet_x0_25_msmt17.pt)."
        )
    return abs_path


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class ReidEmbedder:
    """OSNet-based person embedder.

    All preprocessing (BGR→RGB, resize, normalize) lives here so callers
    just hand over raw uint8 BGR crops the same shape as the YOLO frame.

    Thread-safety: the underlying torch model is not thread-safe. Caller must
    not invoke embed_batch concurrently from multiple threads.
    """

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

        # Pre-build the normalization tensors as buffers on the target device
        # so each embed_batch call avoids the host→device copy of mean/std.
        self._mean = torch.tensor(REID_PIXEL_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std  = torch.tensor(REID_PIXEL_STD,  device=self.device).view(1, 3, 1, 1)
        self._target_h, self._target_w = REID_INPUT_SIZE_HW

        print(f"[INFO] ReidEmbedder ready on {self.device} (model={REID_MODEL_NAME})")

    @torch.no_grad()
    def embed_batch(self, crops_bgr: Iterable[np.ndarray]) -> Optional[torch.Tensor]:
        """Embed N person bbox crops in one forward pass.

        Args:
            crops_bgr: iterable of HxWx3 uint8 BGR arrays (any size, will be
                resized to REID_INPUT_SIZE_HW).

        Returns:
            (N, 512) L2-normalized float tensor on self.device, or None when
            the input iterable is empty.
        """
        crops = [c for c in crops_bgr if c is not None and c.size > 0]
        if not crops:
            return None

        # Resize each crop to OSNet's expected (256, 128) BEFORE stacking — they
        # may all differ in shape. cv2.resize uses INTER_LINEAR by default which
        # is the same as torchreid's default transform.
        resized = [
            cv2.resize(c, (self._target_w, self._target_h), interpolation=cv2.INTER_LINEAR)
            for c in crops
        ]
        # Stack to NHWC uint8, BGR → RGB, NHWC → NCHW, /255 → float
        batch_np = np.stack(resized, axis=0)[..., ::-1]   # BGR→RGB via channel reverse
        # ::-1 view must be contiguous before from_numpy on Windows; .copy() forces that.
        batch = torch.from_numpy(batch_np.copy()).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = (batch - self._mean) / self._std

        embeddings = self.model(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance between two L2-normalized embeddings.

    Both inputs must be 1-D (512,) or matched 2-D (1, 512). Returns a scalar
    float in [0, 2] — 0 = identical direction, 1 = orthogonal, 2 = opposite.
    """
    a = a.flatten()
    b = b.flatten()
    return float(1.0 - torch.dot(a, b).item())


def update_embedding_ema(old: torch.Tensor, new: torch.Tensor,
                         alpha: float = LOCK_EMBEDDING_EMA_ALPHA) -> torch.Tensor:
    """EMA-update a locked embedding toward a freshly matched detection.

    Re-normalizes after the lerp because the linear combination of two unit
    vectors is generally not unit-length, and we want cosine distance to keep
    behaving like cosine distance.
    """
    mixed = (1.0 - alpha) * old + alpha * new
    return F.normalize(mixed, p=2, dim=0)


# ---------------------------------------------------------------------------
# Bbox crop helper
# ---------------------------------------------------------------------------

def crop_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    """Return frame[y1:y2, x1:x2] clamped to frame bounds, or None if degenerate.

    Caller (selector) gets a (None | HxWx3 uint8 BGR) view into the original
    frame — no copy until the embedder resizes.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
