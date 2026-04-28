from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import SimpleITK as sitk
from fastapi import HTTPException
from pydicom.dataset import FileDataset
from pydicom.errors import InvalidDicomError


def _iter_candidate_files(dicom_dir: Path) -> list[Path]:
    return [path for path in dicom_dir.rglob("*") if path.is_file()]


def _read_header(path: Path) -> FileDataset | None:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except (InvalidDicomError, OSError):
        return None


def _extract_position(ds: FileDataset) -> np.ndarray:
    image_position = getattr(ds, "ImagePositionPatient", None)
    if image_position is None:
        raise ValueError("Missing ImagePositionPatient")
    if len(image_position) < 3:
        raise ValueError("Invalid ImagePositionPatient")
    return np.asarray(
        [float(image_position[0]), float(image_position[1]), float(image_position[2])],
        dtype=np.float64,
    )


def _extract_orientation(
    ds: FileDataset,
) -> tuple[np.ndarray, np.ndarray]:
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if orientation is None or len(orientation) < 6:
        raise ValueError("Missing ImageOrientationPatient")
    row = np.asarray(
        [float(orientation[0]), float(orientation[1]), float(orientation[2])],
        dtype=np.float64,
    )
    col = np.asarray(
        [float(orientation[3]), float(orientation[4]), float(orientation[5])],
        dtype=np.float64,
    )
    row_norm = np.linalg.norm(row)
    col_norm = np.linalg.norm(col)
    if row_norm < 1e-8 or col_norm < 1e-8:
        raise ValueError("Invalid ImageOrientationPatient")
    return row / row_norm, col / col_norm


def _slice_coordinate(position: np.ndarray, normal: np.ndarray) -> float:
    return float(np.dot(position, normal))


def _validate_slice_spacing(slice_coords: list[float]) -> float:
    if len(slice_coords) <= 1:
        return 0.0
    diffs = np.diff(np.asarray(slice_coords, dtype=np.float64))
    if np.any(diffs <= 1e-6):
        raise HTTPException(
            status_code=400,
            detail="No valid DICOM series found: duplicate or unsorted slice positions",
        )
    median_spacing = float(np.median(diffs))
    tolerance = max(0.25, 0.1 * median_spacing)
    max_deviation = float(np.max(np.abs(diffs - median_spacing)))
    if max_deviation > tolerance:
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid DICOM series found: inconsistent slice spacing "
                f"(median={median_spacing:.4f}mm, max_deviation={max_deviation:.4f}mm)"
            ),
        )
    return median_spacing


def _build_nifti_affine_ras(
    *,
    direction: tuple[float, ...],
    spacing: tuple[float, ...],
    origin: tuple[float, ...],
    slice_spacing_mm: float,
) -> np.ndarray:
    direction_m = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    sx = float(spacing[0])
    sy = float(spacing[1])
    sz = float(slice_spacing_mm if slice_spacing_mm > 0 else spacing[2])

    # Volume layout is [H, W, D] == [y, x, z].
    col_h = direction_m[:, 1] * sy
    col_w = direction_m[:, 0] * sx
    col_d = direction_m[:, 2] * sz

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = col_h
    affine_lps[:3, 1] = col_w
    affine_lps[:3, 2] = col_d
    affine_lps[:3, 3] = np.asarray(origin, dtype=np.float64)

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def enforce_single_series(dicom_dir: Path, modality: str) -> Path:
    if not dicom_dir.exists() or not dicom_dir.is_dir():
        raise HTTPException(status_code=400, detail="No valid DICOM series found")

    series_uids: set[str] = set()
    valid_ct_instances = 0
    missing_tag_instances = 0

    modality_expected = modality.strip().upper()
    for file_path in _iter_candidate_files(dicom_dir):
        ds = _read_header(file_path)
        if ds is None:
            continue

        series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        modality_value = str(getattr(ds, "Modality", "")).strip().upper()
        image_position = getattr(ds, "ImagePositionPatient", None)

        if not series_uid or image_position is None:
            missing_tag_instances += 1
            continue
        if modality_value != modality_expected:
            continue

        series_uids.add(series_uid)
        valid_ct_instances += 1

    if len(series_uids) > 1:
        raise HTTPException(status_code=400, detail="Multi-series DICOM not supported")
    if valid_ct_instances == 0:
        if missing_tag_instances > 0:
            raise HTTPException(
                status_code=400,
                detail="No valid DICOM series found: missing required DICOM tags",
            )
        raise HTTPException(status_code=400, detail="No valid DICOM series found")

    return dicom_dir


def read_dicom_series(
    path: Path, *, modality: str = "CT"
) -> tuple[np.ndarray, dict[str, Any]]:
    dicom_dir = enforce_single_series(path, modality)

    slices: list[tuple[float, Path, FileDataset, np.ndarray]] = []
    series_uid: str | None = None
    normal: np.ndarray | None = None

    modality_expected = modality.strip().upper()
    for file_path in _iter_candidate_files(dicom_dir):
        ds = _read_header(file_path)
        if ds is None:
            continue

        ds_series_uid = str(getattr(ds, "SeriesInstanceUID", "")).strip()
        modality_value = str(getattr(ds, "Modality", "")).strip().upper()
        if not ds_series_uid or modality_value != modality_expected:
            continue

        if series_uid is None:
            series_uid = ds_series_uid
        if ds_series_uid != series_uid:
            continue

        try:
            position = _extract_position(ds)
            row, col = _extract_orientation(ds)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="No valid DICOM series found: invalid ImagePositionPatient",
            ) from exc

        current_normal = np.cross(row, col)
        norm = np.linalg.norm(current_normal)
        if norm < 1e-8:
            raise HTTPException(
                status_code=400,
                detail="No valid DICOM series found: invalid ImageOrientationPatient",
            )
        current_normal = current_normal / norm

        if normal is None:
            normal = current_normal
        elif float(np.dot(normal, current_normal)) < 0.0:
            current_normal = -current_normal

        coord = _slice_coordinate(position, current_normal)
        slices.append((coord, file_path, ds, position))

    if not slices:
        raise HTTPException(status_code=400, detail="No valid DICOM series found")

    slices.sort(key=lambda item: item[0])
    slice_coords = [item[0] for item in slices]
    validated_spacing = _validate_slice_spacing(slice_coords)
    ordered_files = [str(item[1]) for item in slices]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered_files)
    try:
        image = reader.Execute()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400, detail="Failed to read DICOM series"
        ) from exc

    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    volume = np.transpose(array_zyx, (1, 2, 0))
    first_ds = slices[0][2]

    spacing = image.GetSpacing()
    direction = image.GetDirection()
    origin = image.GetOrigin()
    slice_spacing_mm = float(validated_spacing if validated_spacing > 0 else spacing[2])
    affine_ras = _build_nifti_affine_ras(
        direction=direction,
        spacing=spacing,
        origin=origin,
        slice_spacing_mm=slice_spacing_mm,
    )
    orientation_row, orientation_col = _extract_orientation(first_ds)
    orientation_normal = np.cross(orientation_row, orientation_col)

    metadata: dict[str, Any] = {
        "series_instance_uid": series_uid,
        "modality": modality_expected,
        "patient_id": str(getattr(first_ds, "PatientID", "")).strip() or None,
        "patient_name": str(getattr(first_ds, "PatientName", "")).strip() or None,
        "study_id": str(getattr(first_ds, "StudyID", "")).strip() or None,
        "study_date": str(getattr(first_ds, "StudyDate", "")).strip() or None,
        "geometry": {
            "shape": [int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])],
            "spacing_xyz_mm": [
                float(spacing[0]),
                float(spacing[1]),
                float(spacing[2]),
            ],
            "origin_xyz": [float(origin[0]), float(origin[1]), float(origin[2])],
            "direction": [float(x) for x in direction],
            "slice_spacing_mm": slice_spacing_mm,
            "affine_ras": [[float(value) for value in row] for row in affine_ras],
            "slice_normal": [
                float(orientation_normal[0]),
                float(orientation_normal[1]),
                float(orientation_normal[2]),
            ],
        },
    }
    return volume, metadata
