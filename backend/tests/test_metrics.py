import math

import numpy as np
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image

from services.metrics import compute_volume_metrics


def _write_nifti(path, volume, affine=None):
    if affine is None:
        affine = np.eye(4)
    save(Nifti1Image(volume.astype(np.float32), affine), str(path))


def test_compute_metrics_returns_values_for_aligned_volumes(tmp_path):
    pred_path = tmp_path / "pred.nii.gz"
    ref_path = tmp_path / "ref.nii.gz"
    pred = np.zeros((4, 4, 4), dtype=np.float32)
    ref = np.ones((4, 4, 4), dtype=np.float32)
    _write_nifti(pred_path, pred)
    _write_nifti(ref_path, ref)

    result = compute_volume_metrics(
        pred_path=str(pred_path), reference_path=str(ref_path)
    )

    assert result.status == "completed"
    assert result.psnr is not None
    assert result.ssim is not None
    assert math.isfinite(result.psnr)
    assert result.psnr == 0.0
    assert 0.0 <= result.ssim <= 1.0


def test_compute_metrics_skips_on_shape_mismatch(tmp_path):
    pred_path = tmp_path / "pred.nii.gz"
    ref_path = tmp_path / "ref.nii.gz"
    _write_nifti(pred_path, np.zeros((4, 4, 4), dtype=np.float32))
    _write_nifti(ref_path, np.zeros((4, 4, 3), dtype=np.float32))

    result = compute_volume_metrics(
        pred_path=str(pred_path), reference_path=str(ref_path)
    )

    assert result.status == "skipped"
    assert result.reason == "shape_mismatch"


def test_compute_metrics_skips_on_affine_mismatch(tmp_path):
    pred_path = tmp_path / "pred.nii.gz"
    ref_path = tmp_path / "ref.nii.gz"
    pred_affine = np.eye(4)
    ref_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    _write_nifti(pred_path, np.zeros((4, 4, 4), dtype=np.float32), pred_affine)
    _write_nifti(ref_path, np.zeros((4, 4, 4), dtype=np.float32), ref_affine)

    result = compute_volume_metrics(
        pred_path=str(pred_path), reference_path=str(ref_path)
    )

    assert result.status == "skipped"
    assert result.reason == "affine_mismatch"


def test_compute_metrics_skips_when_reference_missing(tmp_path):
    pred_path = tmp_path / "pred.nii.gz"
    _write_nifti(pred_path, np.zeros((4, 4, 4), dtype=np.float32))

    result = compute_volume_metrics(pred_path=str(pred_path), reference_path=None)

    assert result.status == "skipped"
    assert result.reason == "reference_pet_missing"
