"""
TDD Tests for NIfTI Upload Endpoint

These tests cover:
- Uncompressed .nii uploads
- Compressed .nii.gz uploads
- Invalid file uploads (should return 400, not 500)
- StudyManifest contract validation
- Malformed manifest state error handling

All tests are expected to FAIL initially (RED phase) until the fix is implemented.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    PositronEmissionTomographyImageStorage,
    generate_uid,
)
from fastapi.testclient import TestClient

import main
from main import app, InferenceStatus, MetricsState, StudyManifest, study_manifests


@pytest.fixture
def client(mock_converter, monkeypatch):
    """FastAPI test client fixture."""
    monkeypatch.setattr(main, "converter", mock_converter)
    monkeypatch.setattr(main, "model_loaded", True)
    monkeypatch.setattr(main, "get_converter", lambda _model_path=None: mock_converter)
    return TestClient(app)


@pytest.fixture
def minimal_nifti_bytes(tmp_path):
    """
    Generate minimal valid uncompressed NIfTI file as bytes.
    Creates a 16x16x4 volume (minimum size for model inference).
    Model requires 16x16 spatial dims to avoid max_pool2d errors.
    """
    volume = np.random.rand(16, 16, 4).astype(np.float32)
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    # Write to temp file then read bytes
    temp_file = tmp_path / "temp.nii"
    save(nifti_img, str(temp_file))
    return temp_file.read_bytes()


@pytest.fixture
def minimal_nifti_gz_bytes(tmp_path):
    """
    Generate minimal valid compressed NIfTI (.nii.gz) file as bytes.
    Creates a 16x16x4 volume (minimum size for model inference).
    Model requires 16x16 spatial dims to avoid max_pool2d errors.
    """
    volume = np.random.rand(16, 16, 4).astype(np.float32)
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    # Write to temp file with .gz extension (nibabel handles compression)
    temp_file = tmp_path / "temp.nii.gz"
    save(nifti_img, str(temp_file))
    return temp_file.read_bytes()


@pytest.fixture
def invalid_file_bytes():
    """Generate invalid bytes (not a real NIfTI file)."""
    return b"This is not a valid NIfTI file, just random text."


@pytest.fixture
def mismatched_spacing_pet_bytes(tmp_path):
    volume = np.random.rand(8, 8, 2).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nifti_img = Nifti1Image(volume, affine)
    temp_file = tmp_path / "pet_mismatch.nii.gz"
    save(nifti_img, str(temp_file))
    return temp_file.read_bytes()


@pytest.fixture
def tiny_pet_bytes(tmp_path):
    volume = np.random.rand(2, 2, 1).astype(np.float32)
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)
    temp_file = tmp_path / "pet_tiny.nii.gz"
    save(nifti_img, str(temp_file))
    return temp_file.read_bytes()


def _create_dicom_slice(
    series_uid: str,
    z_position: float,
    instance_number: int,
    *,
    include_patient_tags: bool = True,
    modality: str = "CT",
) -> bytes:
    file_meta = FileMetaDataset()
    modality_upper = modality.strip().upper()
    file_meta.MediaStorageSOPClassUID = (
        PositronEmissionTomographyImageStorage
        if modality_upper == "PT"
        else CTImageStorage
    )
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(
        filename_or_obj="",
        dataset={},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.Modality = modality_upper
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    if include_patient_tags:
        ds.PatientID = "PT-001"
        ds.PatientName = "Unit^Test"
        ds.StudyID = "STD-001"
        ds.StudyDate = datetime.utcnow().strftime("%Y%m%d")
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, float(z_position)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
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

    pixel_array = (np.ones((16, 16), dtype=np.int16) * instance_number).astype(np.int16)
    ds.PixelData = pixel_array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    buffer = io.BytesIO()
    ds.save_as(buffer, write_like_original=False)
    return buffer.getvalue()


@pytest.fixture
def dicom_zip_bytes() -> bytes:
    return b"not-used"


@pytest.fixture
def dicom_zip_missing_patient_tags() -> bytes:
    return b"not-used"


@pytest.fixture
def multi_series_dicom_zip_bytes() -> bytes:
    return b"not-used"


@pytest.fixture
def dicom_directory_upload_parts() -> list[tuple[str, tuple[str, bytes, str]]]:
    series_uid = generate_uid()
    parts: list[tuple[str, tuple[str, bytes, str]]] = []
    for index, z in enumerate([0.0, 1.25, 2.5, 3.75], start=1):
        dicom_bytes = _create_dicom_slice(series_uid, z, index)
        parts.append(
            (
                "dicom_files",
                (
                    f"study/subdir/slice_{index}.dcm",
                    dicom_bytes,
                    "application/dicom",
                ),
            )
        )
    return parts


@pytest.fixture
def pet_dicom_directory_upload_parts() -> list[tuple[str, tuple[str, bytes, str]]]:
    series_uid = generate_uid()
    parts: list[tuple[str, tuple[str, bytes, str]]] = []
    for index, z in enumerate([0.0, 1.25, 2.5, 3.75], start=1):
        dicom_bytes = _create_dicom_slice(series_uid, z, index, modality="PT")
        parts.append(
            (
                "real_pet_dicom_files",
                (
                    f"pet/subdir/slice_{index}.dcm",
                    dicom_bytes,
                    "application/dicom",
                ),
            )
        )
    return parts


def test_upload_uncompressed_nii_succeeds(client, minimal_nifti_bytes):
    """
    Test uploading a valid uncompressed .nii file.
    Expected to FAIL: Current backend may not handle .nii correctly.
    """
    response = client.post(
        "/upload",
        files={
            "ct_file": ("ct_scan.nii", minimal_nifti_bytes, "application/octet-stream")
        },
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["success"] is True
    assert "job_id" in data
    assert data["num_slices"] == 4  # Our test volume is 16x16x4
    assert data["shape"] == [16, 16, 4]
    assert data["has_real_pet"] is False
    manifest = study_manifests[data["job_id"]]
    assert manifest.ct_volume_path.endswith("ct.nii.gz")

    manifest = study_manifests[data["job_id"]]
    assert manifest.inference_status.state == "completed"
    assert manifest.inference_status.started_at is not None
    assert manifest.inference_status.completed_at is not None
    assert manifest.metrics.inference_time_ms is not None
    assert manifest.metrics.inference_time_ms >= 0.0
    assert manifest.metrics.output_shape == (16, 16, 4)
    assert manifest.metrics.slices_processed == 4


def test_upload_compressed_nii_gz_succeeds(client, minimal_nifti_gz_bytes):
    """
    Test uploading a valid compressed .nii.gz file.
    Expected to PASS: This is the currently supported format.
    """
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "ct_scan.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["success"] is True
    assert "job_id" in data
    assert data["num_slices"] == 4
    assert data["shape"] == [16, 16, 4]
    assert data["has_real_pet"] is False


def test_upload_invalid_file_returns_400_not_500(client, invalid_file_bytes):
    """
    Test uploading invalid bytes with .nii.gz extension.
    Expected to FAIL: Current backend returns 500 instead of 400.

    This test verifies that the error handling properly catches
    nibabel load errors and returns HTTP 400 (Bad Request) instead
    of HTTP 500 (Internal Server Error).
    """
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "corrupt.nii.gz",
                invalid_file_bytes,
                "application/octet-stream",
            )
        },
    )

    # Should return 400 Bad Request for invalid file format
    assert response.status_code == 400, (
        f"Expected 400 Bad Request for invalid file, got {response.status_code}. "
        f"Response: {response.text}"
    )
    data = response.json()
    assert "detail" in data
    # Detail should mention invalid format or corrupt file
    assert any(
        keyword in data["detail"].lower()
        for keyword in ["invalid", "corrupt", "format", "nifti"]
    ), f"Expected error detail about invalid NIfTI, got: {data['detail']}"


def test_upload_returns_503_when_model_not_available(monkeypatch, minimal_nifti_bytes):
    monkeypatch.setattr(main, "converter", None)
    monkeypatch.setattr(main, "model_loaded", False)
    monkeypatch.setattr(
        main,
        "get_converter",
        lambda _model_path=None: (_ for _ in ()).throw(
            RuntimeError("checkpoint load failed")
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={
                "ct_file": (
                    "ct_scan.nii",
                    minimal_nifti_bytes,
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 503, response.text
    detail = response.json().get("detail", "")
    assert "model failed to load" in detail.lower()


def test_upload_with_both_ct_and_real_pet(client, minimal_nifti_gz_bytes):
    """
    Test uploading both CT and real PET files (evaluation mode).
    Expected to PASS: This is a baseline test for dual-file upload.
    """
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "ct.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
            "real_pet_file": (
                "pet.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["success"] is True
    assert data["has_real_pet"] is True
    manifest = study_manifests[data["job_id"]]
    assert manifest.ct_volume_path.endswith("ct.nii.gz")
    assert manifest.real_pet_volume_path is not None
    assert manifest.real_pet_volume_path.endswith("real_pet.nii.gz")


def test_upload_uncompressed_real_pet_succeeds(
    client, minimal_nifti_bytes, minimal_nifti_gz_bytes
):
    """
    Test uploading CT as .nii.gz and real PET as uncompressed .nii.
    Expected to FAIL: Backend may not handle uncompressed real_pet.
    """
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "ct.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
            "real_pet_file": (
                "pet.nii",
                minimal_nifti_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["success"] is True
    assert data["has_real_pet"] is True


def test_case_endpoints_work_after_upload(client, minimal_nifti_bytes):
    response = client.post(
        "/upload",
        files={
            "ct_file": ("ct_scan.nii", minimal_nifti_bytes, "application/octet-stream")
        },
    )
    assert response.status_code == 200, response.text

    upload_data = response.json()
    job_id = upload_data["job_id"]

    meta_response = client.get(f"/cases/{job_id}/meta")
    assert meta_response.status_code == 200, meta_response.text
    meta_data = meta_response.json()
    assert meta_data["success"] is True
    assert meta_data["job_id"] == job_id
    assert meta_data["num_slices"] == upload_data["num_slices"]
    assert meta_data["shape"] == upload_data["shape"]
    assert meta_data["source_type"] == "nifti"
    assert meta_data["upload_mode"] == "inference_only"
    assert meta_data["modality"] == "CT"
    assert meta_data["processing_status"] == "completed"
    assert meta_data["spacing_xyz_mm"] is not None
    assert len(meta_data["spacing_xyz_mm"]) == 3
    assert meta_data["slice_spacing_mm"] is None
    assert meta_data["patient_id"] is None
    assert meta_data["patient_name"] is None
    assert meta_data["study_id"] is None
    assert meta_data["study_date"] is None

    slice_response = client.get(f"/cases/{job_id}/slice/0", params={"view": "ct"})
    assert slice_response.status_code == 200
    assert slice_response.headers["content-type"] == "image/png"
    assert slice_response.content.startswith(b"\x89PNG")


def test_study_status_endpoint_returns_manifest_backed_payload(
    client, minimal_nifti_bytes
):
    response = client.post(
        "/upload",
        files={
            "ct_file": ("ct_scan.nii", minimal_nifti_bytes, "application/octet-stream")
        },
    )
    assert response.status_code == 200, response.text
    upload_data = response.json()
    job_id = upload_data["job_id"]

    status_response = client.get(f"/studies/{job_id}/status")
    assert status_response.status_code == 200, status_response.text
    status_data = status_response.json()
    assert status_data["success"] is True
    assert status_data["study_id"] == job_id
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "completed"
    assert status_data["num_slices"] == upload_data["num_slices"]
    assert status_data["shape"] == upload_data["shape"]
    assert status_data["metrics"]["inference_time_ms"] is not None
    assert status_data["metrics"]["slices_processed"] == upload_data["num_slices"]


def test_study_result_endpoint_returns_frontend_ready_payload(
    client, minimal_nifti_bytes
):
    response = client.post(
        "/upload",
        files={
            "ct_file": ("ct_scan.nii", minimal_nifti_bytes, "application/octet-stream")
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    result_response = client.get(f"/studies/{job_id}/result")
    assert result_response.status_code == 200, result_response.text
    result_data = result_response.json()
    assert result_data["success"] is True
    assert result_data["study_id"] == job_id
    assert result_data["job_id"] == job_id
    assert result_data["status"] == "completed"
    assert result_data["ct"]["available"] is True
    assert result_data["predicted_pet"]["available"] is True
    assert result_data["real_pet"]["available"] is False
    assert result_data["ct"]["nifti_path"].startswith("/uploads/")
    assert "\\" not in result_data["ct"]["nifti_path"]
    assert result_data["predicted_pet"]["nifti_path"].startswith("/outputs/")
    assert "\\" not in result_data["predicted_pet"]["nifti_path"]
    assert result_data["real_pet"]["nifti_path"] is None
    assert result_data["ct"]["slice_endpoint_template"] == (
        f"/cases/{job_id}/slice/{{index}}?view=ct"
    )
    assert result_data["predicted_pet"]["slice_endpoint_template"] == (
        f"/cases/{job_id}/slice/{{index}}?view=pred_pet"
    )
    assert result_data["real_pet"]["slice_endpoint_template"] is None

    ct_volume_response = client.get(result_data["ct"]["nifti_path"])
    assert ct_volume_response.status_code == 200
    assert len(ct_volume_response.content) > 0

    pred_volume_response = client.get(result_data["predicted_pet"]["nifti_path"])
    assert pred_volume_response.status_code == 200
    assert len(pred_volume_response.content) > 0


@pytest.mark.parametrize("endpoint", ["status", "result"])
def test_study_endpoints_return_404_for_missing_study(client, endpoint):
    response = client.get(f"/studies/missing-study-id/{endpoint}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Study not found"


def test_dicom_zip_upload_rejected(client, dicom_zip_bytes):
    response = client.post(
        "/upload",
        files={"ct_file": ("ct-study.zip", dicom_zip_bytes, "application/zip")},
    )

    assert response.status_code == 400, response.text
    assert "zip" in response.json().get("detail", "").lower()


def test_dicom_zip_upload_rejected_with_optional_real_pet(
    client, dicom_zip_bytes, minimal_nifti_gz_bytes
):
    response = client.post(
        "/upload",
        files={
            "ct_file": ("ct-study.zip", dicom_zip_bytes, "application/zip"),
            "real_pet_file": (
                "pet.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 400, response.text
    assert "zip" in response.json().get("detail", "").lower()


def test_study_endpoints_allow_lookup_by_manifest_study_id(
    client, dicom_directory_upload_parts
):
    response = client.post("/upload", files=dicom_directory_upload_parts)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    manifest = study_manifests[job_id]
    assert manifest.study_id is not None

    status_response = client.get(f"/studies/{manifest.study_id}/status")
    assert status_response.status_code == 200, status_response.text
    status_data = status_response.json()
    assert status_data["study_id"] == manifest.study_id
    assert status_data["job_id"] == job_id

    result_response = client.get(f"/studies/{manifest.study_id}/result")
    assert result_response.status_code == 200, result_response.text
    result_data = result_response.json()
    assert result_data["study_id"] == manifest.study_id
    assert result_data["job_id"] == job_id


def test_case_meta_handles_missing_optional_tags(client, dicom_directory_upload_parts):
    response = client.post("/upload", files=dicom_directory_upload_parts)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    manifest = study_manifests[job_id]
    assert manifest.study_id is not None

    meta_response = client.get(f"/cases/{job_id}/meta")
    assert meta_response.status_code == 200, meta_response.text
    meta_data = meta_response.json()
    assert meta_data["success"] is True
    assert meta_data["source_type"] == "dicom_dir"
    assert meta_data["upload_mode"] == "inference_only"
    assert meta_data["modality"] == "CT"
    assert meta_data["processing_status"] == "completed"
    assert meta_data["spacing_xyz_mm"] is not None
    assert len(meta_data["spacing_xyz_mm"]) == 3
    assert meta_data["slice_spacing_mm"] is not None


def test_dicom_directory_upload_succeeds_with_repeated_parts(
    client, dicom_directory_upload_parts
):
    response = client.post("/upload", files=dicom_directory_upload_parts)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["source_type"] == "dicom_dir"
    assert data["has_real_pet"] is False
    assert data["num_slices"] == 4

    job_id = data["job_id"]
    manifest = study_manifests[job_id]
    assert manifest.source_type == "dicom_dir"
    assert manifest.upload_mode == "inference_only"
    assert manifest.ct_volume_path.endswith("ct.nii.gz")


def test_dicom_directory_upload_succeeds_with_optional_real_pet(
    client, dicom_directory_upload_parts, minimal_nifti_gz_bytes
):
    response = client.post(
        "/upload",
        files=dicom_directory_upload_parts
        + [
            (
                "real_pet_file",
                (
                    "pet.nii.gz",
                    minimal_nifti_gz_bytes,
                    "application/octet-stream",
                ),
            )
        ],
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["source_type"] == "dicom_dir"
    assert data["has_real_pet"] is True

    manifest = study_manifests[data["job_id"]]
    assert manifest.real_pet_volume_path is not None
    assert manifest.real_pet_volume_path.endswith("real_pet.nii.gz")
    assert manifest.upload_mode == "with_evaluation"


def test_dicom_directory_upload_succeeds_with_optional_pet_dicom_folder(
    client,
    dicom_directory_upload_parts,
    pet_dicom_directory_upload_parts,
):
    response = client.post(
        "/upload",
        files=dicom_directory_upload_parts + pet_dicom_directory_upload_parts,
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["success"] is True
    assert data["source_type"] == "dicom_dir"
    assert data["has_real_pet"] is True

    manifest = study_manifests[data["job_id"]]
    assert manifest.real_pet_volume_path is not None
    assert manifest.real_pet_volume_path.endswith("real_pet.nii.gz")
    assert manifest.upload_mode == "with_evaluation"


def test_multi_series_dicom_rejected(client, multi_series_dicom_zip_bytes):
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "multi-series.zip",
                multi_series_dicom_zip_bytes,
                "application/zip",
            )
        },
    )

    assert response.status_code == 400, response.text
    assert "zip" in response.json().get("detail", "").lower()


def test_upload_real_pet_alignment_metadata_recorded(
    client, minimal_nifti_gz_bytes, mismatched_spacing_pet_bytes
):
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "ct.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
            "real_pet_file": (
                "pet.nii.gz",
                mismatched_spacing_pet_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    manifest = study_manifests[job_id]
    assert manifest.real_pet_volume_path is not None
    assert manifest.real_pet_volume_path.endswith("real_pet.nii.gz")
    assert manifest.geometry is not None
    assert manifest.geometry["reference_pet"]["alignment"]["status"] in {
        "already_aligned",
        "resampled",
    }


def test_upload_rejects_incompatible_real_pet_geometry(
    client, minimal_nifti_gz_bytes, tiny_pet_bytes
):
    response = client.post(
        "/upload",
        files={
            "ct_file": (
                "ct.nii.gz",
                minimal_nifti_gz_bytes,
                "application/octet-stream",
            ),
            "real_pet_file": (
                "pet_tiny.nii.gz",
                tiny_pet_bytes,
                "application/octet-stream",
            ),
        },
    )

    assert response.status_code == 400, response.text
    detail = response.json().get("detail", "")
    assert "geometry incompatible" in detail.lower()


# ====== NEW MANIFEST CONTRACT TESTS ======


def test_study_manifest_model_validates_required_fields():
    """
    Test that StudyManifest validates all required fields.
    This test verifies the manifest contract shape is correct.
    """
    manifest_data = {
        "job_id": "test-job-001",
        "source_type": "nifti",
        "upload_mode": "inference_only",
        "ct_volume_path": "/uploads/test-job-001/ct.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-001/pred_pet.nii.gz",
        "num_slices": 128,
        "shape": (512, 512, 128),
    }

    # Should not raise validation error
    manifest = StudyManifest(**manifest_data)

    # Verify all required fields exist and have correct values
    assert manifest.job_id == "test-job-001"
    assert manifest.source_type == "nifti"
    assert manifest.upload_mode == "inference_only"
    assert manifest.ct_volume_path == "/uploads/test-job-001/ct.nii.gz"
    assert manifest.pred_pet_volume_path == "/outputs/test-job-001/pred_pet.nii.gz"
    assert manifest.num_slices == 128
    assert manifest.shape == (512, 512, 128)


def test_study_manifest_has_inference_status():
    """
    Test that StudyManifest includes inference_status field with default state.
    """
    manifest_data = {
        "job_id": "test-job-002",
        "source_type": "nifti",
        "upload_mode": "inference_only",
        "ct_volume_path": "/uploads/test-job-002/ct.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-002/pred_pet.nii.gz",
        "num_slices": 64,
        "shape": (256, 256, 64),
    }

    manifest = StudyManifest(**manifest_data)

    # Verify inference_status exists and is properly initialized
    assert hasattr(manifest, "inference_status")
    assert isinstance(manifest.inference_status, InferenceStatus)
    assert manifest.inference_status.state == "pending"
    assert manifest.inference_status.started_at is None
    assert manifest.inference_status.error is None


def test_study_manifest_has_metrics_state():
    """
    Test that StudyManifest includes metrics field with default empty state.
    """
    manifest_data = {
        "job_id": "test-job-003",
        "source_type": "nifti",
        "upload_mode": "with_evaluation",
        "ct_volume_path": "/uploads/test-job-003/ct.nii.gz",
        "real_pet_volume_path": "/uploads/test-job-003/real_pet.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-003/pred_pet.nii.gz",
        "num_slices": 100,
        "shape": (512, 512, 100),
    }

    manifest = StudyManifest(**manifest_data)

    # Verify metrics field exists and is properly initialized
    assert hasattr(manifest, "metrics")
    assert isinstance(manifest.metrics, MetricsState)
    assert manifest.metrics.inference_time_ms is None
    assert manifest.metrics.output_shape is None
    assert manifest.metrics.slices_processed is None
    assert manifest.metrics.psnr is None
    assert manifest.metrics.ssim is None
    assert manifest.metrics.evaluation_status is None
    assert manifest.metrics.evaluation_reason is None


def test_study_manifest_supports_evaluation_mode():
    """
    Test that StudyManifest correctly tracks evaluation mode with real_pet_volume_path.
    """
    manifest_data = {
        "job_id": "test-job-004",
        "source_type": "nifti",
        "upload_mode": "with_evaluation",
        "ct_volume_path": "/uploads/test-job-004/ct.nii.gz",
        "real_pet_volume_path": "/uploads/test-job-004/real_pet.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-004/pred_pet.nii.gz",
        "num_slices": 80,
        "shape": (256, 256, 80),
    }

    manifest = StudyManifest(**manifest_data)

    assert manifest.upload_mode == "with_evaluation"
    assert manifest.real_pet_volume_path == "/uploads/test-job-004/real_pet.nii.gz"


def test_study_manifest_missing_required_field_fails():
    """
    Test that StudyManifest validation fails when required fields are missing.
    This is a negative test for malformed manifest state.
    """
    incomplete_data: dict[str, Any] = {
        "job_id": "test-job-005",
        # Missing: source_type, upload_mode, ct_volume_path, pred_pet_volume_path, num_slices, shape
        "ct_volume_path": "/uploads/test-job-005/ct.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-005/pred_pet.nii.gz",
        # Missing: num_slices and shape
    }

    # Should raise validation error due to missing required fields
    with pytest.raises(Exception) as exc_info:
        StudyManifest(**incomplete_data)

    # Verify that validation error mentions missing fields
    error_message = str(exc_info.value)
    assert (
        "num_slices" in error_message
        or "shape" in error_message
        or "field" in error_message.lower()
    )


def test_study_manifest_invalid_source_type_fails():
    """
    Test that StudyManifest validation fails with invalid source_type.
    Demonstrates controlled error handling for malformed manifest state.
    """
    invalid_data = {
        "job_id": "test-job-006",
        "source_type": "invalid_type",  # Not in allowed Literal values
        "upload_mode": "inference_only",
        "ct_volume_path": "/uploads/test-job-006/ct.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-006/pred_pet.nii.gz",
        "num_slices": 50,
        "shape": (128, 128, 50),
    }

    # Should raise validation error
    with pytest.raises(Exception) as exc_info:
        StudyManifest(**invalid_data)

    # Verify validation caught the invalid source_type
    error_message = str(exc_info.value)
    assert "source_type" in error_message.lower() or "valid" in error_message.lower()


def test_study_manifest_serializes_to_json():
    """
    Test that StudyManifest can be serialized to JSON for API responses.
    """
    manifest_data = {
        "job_id": "test-job-007",
        "source_type": "nifti",
        "upload_mode": "inference_only",
        "ct_volume_path": "/uploads/test-job-007/ct.nii.gz",
        "pred_pet_volume_path": "/outputs/test-job-007/pred_pet.nii.gz",
        "num_slices": 75,
        "shape": (384, 384, 75),
    }

    manifest = StudyManifest(**manifest_data)

    # Should serialize without errors
    json_data = manifest.model_dump_json()
    assert isinstance(json_data, str)

    # Verify key fields are in the JSON
    import json

    parsed = json.loads(json_data)
    assert parsed["job_id"] == "test-job-007"
    assert parsed["num_slices"] == 75
    assert parsed["shape"] == [384, 384, 75]
