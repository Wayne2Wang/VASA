"""Shared BGR mask overlay (dimmed base + color blend) for viz and agent working-mask dumps."""
from __future__ import annotations

import numpy as np

# OpenCV BGR: green highlight on dimmed base (final overlay.png and working-mask frames).
DEFAULT_MASK_OVERLAY_BGR: tuple[int, int, int] = (0, 255, 0)
DEFAULT_MASK_OVERLAY_ALPHA: float = 0.58
DEFAULT_MASK_OVERLAY_IMAGE_DIM: float = 0.48


def _collapse_mask_to_2d(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask array to H×W.

    ``cv2.imread`` loads PNG as BGR; taking only ``[..., 0]`` is the **blue** channel.
    Red- or green-only RGB masks would largely vanish. We use the per-pixel max across
    channels (and alpha when present) so any channel that marks foreground is kept.
    """
    m = np.asarray(mask)
    if m.ndim == 2:
        return m
    if m.ndim != 3:
        raise ValueError(f"mask must be 2D or 3D H×W×C, got shape {m.shape}")
    if m.shape[2] == 1:
        return m[:, :, 0]
    return np.max(m, axis=2)


def _resize_mask_nearest(mask_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize so mask aligns with ``image_bgr``."""
    if mask_2d.shape[:2] == (height, width):
        return mask_2d
    from PIL import Image

    a = np.asarray(mask_2d)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    out = np.asarray(Image.fromarray(a).resize((width, height), Image.NEAREST))
    if mask_2d.dtype == bool:
        return out > 127
    return out


def mask_to_binary_u8(mask: np.ndarray, *, uint8_gt: int = 0) -> np.ndarray:
    """Normalize masks to 0/1 uint8 (GT 0/1, pred PNGs 0–255, floats, bool).

    For integer masks whose max is **greater than 1** (typical 8-bit PNG), a pixel is
    foreground iff ``m > uint8_gt``. Default ``uint8_gt=0`` keeps any non-zero
    value (soft / anti-aliased edges). Use ``uint8_gt=127`` for the older strict
    cutoff (only values 128–255).

    Multi-channel inputs (e.g. BGR/BGRA from OpenCV) are collapsed with a per-pixel
    maximum across channels before thresholding.
    """
    m = _collapse_mask_to_2d(mask)
    if m.size == 0:
        return m.astype(np.uint8)
    if m.dtype.kind == "f":
        return (m > 0.5).astype(np.uint8)
    if m.dtype.kind == "b":
        return m.astype(np.uint8)
    mx = int(m.max())
    if mx <= 1:
        return (m > 0).astype(np.uint8)
    return (m > uint8_gt).astype(np.uint8)


def overlay_mask_bgr(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float,
    color_bgr: tuple[int, int, int],
    image_dim: float,
    mask_uint8_gt: int = 0,
) -> np.ndarray:
    """Dim image by ``image_dim``, then blend ``color_bgr`` on foreground with strength ``alpha``.

    ``mask_uint8_gt`` is passed to :func:`mask_to_binary_u8` for 8-bit masks (see there).
    """
    base = (image_bgr.astype(np.float32) * float(image_dim)).clip(0, 255).astype(np.uint8)
    m2d = _collapse_mask_to_2d(mask)
    hi, wi = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    if m2d.shape[:2] != (hi, wi):
        m2d = _resize_mask_nearest(m2d, hi, wi)
    m = mask_to_binary_u8(m2d, uint8_gt=mask_uint8_gt)
    overlay = np.zeros_like(base)
    overlay[:] = color_bgr
    blend = (overlay.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
    m3 = m[:, :, np.newaxis]
    return np.where(m3 > 0, blend, base)
