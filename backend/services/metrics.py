from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from utils.image_processing import ImageProcessor


@dataclass
class VolumeMetricsResult:
    status: Literal["completed", "skipped", "failed"]
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    reason: Optional[str] = None


def _geometry_compatible(
    pred_shape: tuple[int, int, int],
    ref_shape: tuple[int, int, int],
    pred_affine: np.ndarray,
    ref_affine: np.ndarray,
) -> Optional[str]:
    if pred_shape != ref_shape:
        return "shape_mismatch"
    if not np.allclose(pred_affine, ref_affine, atol=1e-4):
        return "affine_mismatch"
    return None


def _compute_psnr(mse: float, data_range: float) -> float:
    if mse <= 0.0:
        return float("inf")
    if data_range <= 0.0:
        return 0.0
    return 20.0 * float(np.log10(data_range)) - 10.0 * float(np.log10(mse))


def _compute_ssim(pred: np.ndarray, ref: np.ndarray, data_range: float) -> float:
    pred = pred.astype(np.float64, copy=False)
    ref = ref.astype(np.float64, copy=False)

    if data_range <= 0.0:
        return 1.0 if np.allclose(pred, ref) else 0.0

    mu_pred = float(np.mean(pred))
    mu_ref = float(np.mean(ref))
    sigma_pred = float(np.mean((pred - mu_pred) ** 2))
    sigma_ref = float(np.mean((ref - mu_ref) ** 2))
    sigma_cross = float(np.mean((pred - mu_pred) * (ref - mu_ref)))

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    numerator = (2.0 * mu_pred * mu_ref + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred**2 + mu_ref**2 + c1) * (sigma_pred + sigma_ref + c2)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_volume_metrics(
    *, pred_path: str, reference_path: Optional[str]
) -> VolumeMetricsResult:
    if reference_path is None:
        return VolumeMetricsResult(status="skipped", reason="reference_pet_missing")

    processor = ImageProcessor()
    try:
        pred_volume, pred_affine = processor.load_nifti(pred_path)
        ref_volume, ref_affine = processor.load_nifti(reference_path)
    except Exception as exc:
        return VolumeMetricsResult(
            status="failed",
            reason=f"load_failed: {exc}",
        )

    mismatch_reason = _geometry_compatible(
        pred_volume.shape,
        ref_volume.shape,
        pred_affine,
        ref_affine,
    )
    if mismatch_reason is not None:
        return VolumeMetricsResult(status="skipped", reason=mismatch_reason)

    if not np.isfinite(pred_volume).all() or not np.isfinite(ref_volume).all():
        return VolumeMetricsResult(status="failed", reason="non_finite_values")

    diff = pred_volume.astype(np.float64) - ref_volume.astype(np.float64)
    mse = float(np.mean(diff**2))
    data_range = float(np.max(ref_volume) - np.min(ref_volume))
    psnr = _compute_psnr(mse, data_range)
    ssim = _compute_ssim(pred_volume, ref_volume, data_range)
    return VolumeMetricsResult(status="completed", psnr=psnr, ssim=ssim)
