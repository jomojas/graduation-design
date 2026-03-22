import io
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from fastapi import HTTPException
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from utils.dicom_ingest import read_dicom_series
from utils.standardization import (
    align_reference_pet_to_ct,
    standardize_dicom_ct,
)


def _create_dicom_slice(
    series_uid: str,
    z_position: float,
    instance_number: int,
    pixel_value: int,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.PatientID = "PT-001"
    ds.PatientName = "Unit^Test"
    ds.StudyID = "STD-001"
    ds.StudyDate = datetime.utcnow().strftime("%Y%m%d")
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, float(z_position)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.SliceThickness = float(slice_thickness)
    ds.Rows = 16
    ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.PixelData = (np.ones((16, 16), dtype=np.int16) * pixel_value).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    buffer = io.BytesIO()
    ds.save_as(buffer, write_like_original=False)
    return buffer.getvalue()


def _write_series(
    target_dir: Path,
    z_positions: list[float],
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
    slice_thickness: float = 1.0,
) -> None:
    series_uid = generate_uid()
    for index, z in enumerate(z_positions, start=1):
        pixel_value = int(round(z * 10))
        data = _create_dicom_slice(
            series_uid,
            z,
            instance_number=99 - index,
            pixel_value=pixel_value,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
        )
        # Deliberately reverse filename ordering to enforce spatial sorting.
        (target_dir / f"slice_{len(z_positions) - index:02d}.dcm").write_bytes(data)


def test_dicom_read_uses_spatial_order_not_filename(tmp_path: Path):
    dicom_dir = tmp_path / "study"
    dicom_dir.mkdir()
    _write_series(dicom_dir, [4.5, 0.0, 1.5, 3.0])

    volume, metadata = read_dicom_series(dicom_dir)

    assert volume.shape == (16, 16, 4)
    assert np.allclose(volume[:, :, 0], 0.0)
    assert np.allclose(volume[:, :, 1], 15.0)
    assert np.allclose(volume[:, :, 2], 30.0)
    assert np.allclose(volume[:, :, 3], 45.0)
    assert metadata["geometry"]["slice_spacing_mm"] == pytest.approx(1.5, abs=1e-3)


def test_dicom_rejects_inconsistent_slice_spacing(tmp_path: Path):
    dicom_dir = tmp_path / "study_bad_spacing"
    dicom_dir.mkdir()
    _write_series(dicom_dir, [0.0, 1.2, 3.8, 4.9])

    with pytest.raises(HTTPException) as exc_info:
        read_dicom_series(dicom_dir)

    assert exc_info.value.status_code == 400
    assert "inconsistent slice spacing" in str(exc_info.value.detail).lower()


def test_standardize_dicom_ct_exports_canonical_nii_gz(tmp_path: Path):
    dicom_dir = tmp_path / "study_standardize"
    dicom_dir.mkdir()
    _write_series(
        dicom_dir,
        [0.0, 1.5, 3.0, 4.5],
        pixel_spacing=(0.8, 1.2),
        slice_thickness=1.5,
    )

    output_path = tmp_path / "ct.nii.gz"
    result = standardize_dicom_ct(dicom_dir, output_path)

    assert result.ct_path == output_path
    assert output_path.exists()
    nifti = cast(Nifti1Image, load(str(output_path)))
    assert nifti.shape == (16, 16, 4)
    assert result.geometry["shape"] == [16, 16, 4]
    assert nifti.affine is not None
    assert not np.allclose(nifti.affine, np.eye(4), atol=1e-6)

    affine_spacing = np.linalg.norm(nifti.affine[:3, :3], axis=0)
    geom_spacing = result.geometry["spacing_xyz_mm"]
    assert np.allclose(
        affine_spacing,
        [geom_spacing[1], geom_spacing[0], result.geometry["slice_spacing_mm"]],
        atol=1e-3,
    )


def test_reference_pet_is_resampled_to_ct_geometry(tmp_path: Path):
    ct_data = np.ones((16, 16, 4), dtype=np.float32)
    pet_data = np.ones((8, 8, 2), dtype=np.float32) * 2.0
    ct_path = tmp_path / "ct.nii.gz"
    pet_path = tmp_path / "pet_input.nii.gz"
    aligned_path = tmp_path / "pet_aligned.nii.gz"

    save(Nifti1Image(ct_data, np.eye(4)), str(ct_path))
    save(
        Nifti1Image(pet_data, np.diag([2.0, 2.0, 2.0, 1.0])),
        str(pet_path),
    )

    saved_path, meta = align_reference_pet_to_ct(ct_path, pet_path, aligned_path)
    assert saved_path == aligned_path
    assert meta["status"] == "resampled"

    ct_affine = cast(Nifti1Image, load(str(ct_path))).affine
    aligned_img = cast(Nifti1Image, load(str(aligned_path)))
    assert ct_affine is not None
    assert aligned_img.affine is not None
    assert aligned_img.shape == (16, 16, 4)
    assert np.allclose(aligned_img.affine, ct_affine, atol=1e-4)


def test_reference_pet_rejects_extreme_extent_mismatch(tmp_path: Path):
    ct_data = np.ones((64, 64, 32), dtype=np.float32)
    pet_data = np.ones((4, 4, 2), dtype=np.float32)
    ct_path = tmp_path / "ct_large.nii.gz"
    pet_path = tmp_path / "pet_tiny.nii.gz"
    aligned_path = tmp_path / "pet_aligned.nii.gz"

    save(Nifti1Image(ct_data, np.diag([1.0, 1.0, 1.0, 1.0])), str(ct_path))
    save(Nifti1Image(pet_data, np.diag([1.0, 1.0, 1.0, 1.0])), str(pet_path))

    with pytest.raises(HTTPException) as exc_info:
        align_reference_pet_to_ct(ct_path, pet_path, aligned_path)

    assert exc_info.value.status_code == 400
    assert "geometry incompatible" in str(exc_info.value.detail).lower()
