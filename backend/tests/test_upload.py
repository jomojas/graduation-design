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

import gzip
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app, StudyManifest, InferenceStatus, MetricsState


@pytest.fixture
def client():
    """FastAPI test client fixture."""
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
    nifti_img = nib.Nifti1Image(volume, affine)

    # Write to temp file then read bytes
    temp_file = tmp_path / "temp.nii"
    nib.save(nifti_img, str(temp_file))
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
    nifti_img = nib.Nifti1Image(volume, affine)

    # Write to temp file with .gz extension (nibabel handles compression)
    temp_file = tmp_path / "temp.nii.gz"
    nib.save(nifti_img, str(temp_file))
    return temp_file.read_bytes()


@pytest.fixture
def invalid_file_bytes():
    """Generate invalid bytes (not a real NIfTI file)."""
    return b"This is not a valid NIfTI file, just random text."


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

    slice_response = client.get(f"/cases/{job_id}/slice/0", params={"view": "ct"})
    assert slice_response.status_code == 200
    assert slice_response.headers["content-type"] == "image/png"
    assert slice_response.content.startswith(b"\x89PNG")


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
    incomplete_data = {
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
