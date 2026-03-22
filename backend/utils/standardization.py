from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from fastapi import HTTPException
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image

from utils.dicom_ingest import read_dicom_series


@dataclass
class StandardizationResult:
    ct_path: Path
    real_pet_path: Path | None
    geometry: dict[str, Any]
    metadata: dict[str, Any] | None = None


def _safe_axcodes(affine: np.ndarray) -> list[str]:
    try:
        axcodes = nib.orientations.aff2axcodes(affine)
    except Exception:
        return ["?", "?", "?"]
    return [str(code) for code in axcodes]


def extract_nifti_geometry(nifti_path: Path) -> dict[str, Any]:
    image: Any = load(str(nifti_path))
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    if volume.ndim != 3:
        raise HTTPException(
            status_code=400, detail="Only 3D NIfTI volumes are supported"
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    return {
        "shape": [int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])],
        "spacing_xyz_mm": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
        "origin_xyz": [
            float(affine[0, 3]),
            float(affine[1, 3]),
            float(affine[2, 3]),
        ],
        "orientation_codes": _safe_axcodes(affine),
        "affine": [[float(v) for v in row] for row in affine.tolist()],
    }


def standardize_dicom_ct(
    dicom_dir: Path,
    ct_output_path: Path,
) -> StandardizationResult:
    ct_volume, metadata = read_dicom_series(dicom_dir)
    geometry = metadata.get("geometry") or {}
    affine_ras = np.asarray(geometry.get("affine_ras", np.eye(4)), dtype=np.float64)
    # Keep existing model-facing semantics: volume layout is [H, W, D].
    save(Nifti1Image(ct_volume.astype(np.float32), affine_ras), str(ct_output_path))
    geometry["standardized_path"] = str(ct_output_path)
    return StandardizationResult(
        ct_path=ct_output_path,
        real_pet_path=None,
        geometry=geometry,
        metadata=metadata,
    )


def standardize_nifti_to_niigz(input_path: Path, output_path: Path) -> Path:
    image: Any = load(str(input_path))
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError("Only 3D NIfTI volumes are supported")
    affine = np.asarray(image.affine, dtype=np.float64)
    save(Nifti1Image(volume, affine), str(output_path))
    return output_path


def _images_match_geometry(reference: sitk.Image, moving: sitk.Image) -> bool:
    if reference.GetSize() != moving.GetSize():
        return False
    if not np.allclose(reference.GetSpacing(), moving.GetSpacing(), atol=1e-5):
        return False
    if not np.allclose(reference.GetOrigin(), moving.GetOrigin(), atol=1e-5):
        return False
    return np.allclose(reference.GetDirection(), moving.GetDirection(), atol=1e-5)


def _physical_extent_mm(image: sitk.Image) -> float:
    size = np.asarray(image.GetSize(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    return float(np.linalg.norm(size * spacing))


def align_reference_pet_to_ct(
    ct_path: Path,
    pet_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Any]]:
    try:
        ct_img = sitk.ReadImage(str(ct_path))
        pet_img = sitk.ReadImage(str(pet_path))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Reference PET geometry incompatible: failed to read NIfTI ({str(exc)})",
        ) from exc

    if ct_img.GetDimension() != 3 or pet_img.GetDimension() != 3:
        raise HTTPException(
            status_code=400,
            detail="Reference PET geometry incompatible: only 3D NIfTI volumes are supported",
        )

    status = "already_aligned"
    reason = "Input PET already matches CT geometry"
    if not _images_match_geometry(ct_img, pet_img):
        ct_extent = _physical_extent_mm(ct_img)
        pet_extent = _physical_extent_mm(pet_img)
        larger = max(ct_extent, pet_extent)
        smaller = min(ct_extent, pet_extent)
        if larger > 0 and (smaller / larger) < 0.2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Reference PET geometry incompatible: physical extent differs "
                    "too much from CT for reliable alignment"
                ),
            )
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ct_img)
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        pet_img = resampler.Execute(pet_img)
        status = "resampled"
        reason = "Reference PET was resampled to CT geometry"

    sitk.WriteImage(pet_img, str(output_path), useCompression=True)
    return output_path, {"status": status, "reason": reason}
