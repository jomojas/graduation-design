"""
TDD Tests for NIfTI Upload Endpoint

These tests cover:
- Uncompressed .nii uploads
- Compressed .nii.gz uploads
- Invalid file uploads (should return 400, not 500)

All tests are expected to FAIL initially (RED phase) until the fix is implemented.
"""

import gzip
import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app


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
