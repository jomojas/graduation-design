"""
Shared pytest fixtures and mocks for backend tests.

Provides:
- NIfTI file fixtures (valid, invalid, various geometries)
- Mock model fixtures
- ImageProcessor instances
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image


@pytest.fixture
def tmp_nifti_dir(tmp_path):
    """Create a temporary directory for NIfTI test files."""
    return tmp_path / "nifti_files"


@pytest.fixture
def minimal_ct_volume(tmp_nifti_dir):
    """
    Create a minimal valid CT volume (16x16x4).
    Model requires 16x16 spatial dims minimum.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(16, 16, 4).astype(np.float32) * 100 - 50  # HU-like values
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "ct_minimal.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def standard_ct_volume(tmp_nifti_dir):
    """
    Create a standard CT volume (64x64x32) with realistic HU values.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    # HU range: -1000 to +3000, clipped to -900 to +200 typically
    volume = np.random.rand(64, 64, 32).astype(np.float32) * 1100 - 900
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "ct_standard.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def mismatched_pet_volume(tmp_nifti_dir):
    """
    Create a PET volume with different shape than CT.
    Used to test geometry mismatch detection.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(32, 32, 16).astype(np.float32)  # Different shape
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "pet_mismatched.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def mismatched_spacing_volume(tmp_nifti_dir):
    """
    Create a volume with different voxel spacing (affine).
    Used to test spacing mismatch detection.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(64, 64, 32).astype(np.float32)
    # Different voxel spacing: 2x2x2 instead of 1x1x1
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "pet_spacing_mismatch.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def small_mismatched_spacing_pet_volume(tmp_nifti_dir):
    """
    Create a smaller PET volume with mismatched spacing for resampling tests.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(8, 8, 2).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "pet_small_spacing_mismatch.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def tiny_pet_volume(tmp_nifti_dir):
    """
    Create a tiny PET volume for incompatible geometry rejection tests.
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(2, 2, 1).astype(np.float32)
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "pet_tiny.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def invalid_2d_volume(tmp_nifti_dir):
    """
    Create an invalid 2D NIfTI file (should be 3D).
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(64, 64).astype(np.float32)  # 2D only
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "invalid_2d.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def invalid_4d_volume(tmp_nifti_dir):
    """
    Create an invalid 4D NIfTI file (should be 3D).
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(64, 64, 32, 3).astype(np.float32)  # 4D (e.g., RGB)
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "invalid_4d.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def empty_volume(tmp_nifti_dir):
    """
    Create a volume with no slices (invalid for inference).
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(16, 16, 0).astype(np.float32)  # Zero slices
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "empty_volume.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def corrupt_nifti_file(tmp_nifti_dir):
    """
    Create a corrupt NIfTI file (invalid bytes).
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    filepath = tmp_nifti_dir / "corrupt.nii.gz"
    filepath.write_bytes(b"This is not a valid NIfTI file")
    return str(filepath)


@pytest.fixture
def small_volume_16x16x4(tmp_nifti_dir):
    """
    Create the smallest valid volume for model inference (16x16x4).
    """
    tmp_nifti_dir.mkdir(exist_ok=True)
    volume = np.random.rand(16, 16, 4).astype(np.float32) * 100
    affine = np.eye(4)
    nifti_img = Nifti1Image(volume, affine)

    filepath = tmp_nifti_dir / "small_16x16x4.nii.gz"
    save(nifti_img, str(filepath))
    return str(filepath), volume, affine


@pytest.fixture
def mock_model():
    """
    Create a mock PyTorch model that mimics the generator behavior.

    Returns a MagicMock that:
    - Accepts input tensor of shape (batch=1, channels=7, H, W)
    - Returns output of shape (batch=1, channels=2, H, W)
    """
    mock = MagicMock()

    def mock_forward(input_tensor):
        batch, channels, h, w = input_tensor.shape
        # Simulate generator output: [batch, 2, h, w] (CT channel + PET channel)
        output = torch.randn(batch, 2, h, w, dtype=torch.float32)
        return output

    mock.side_effect = mock_forward
    return mock


@pytest.fixture
def mock_converter(mock_model, monkeypatch):
    """
    Create a mocked CT2PETConverter that doesn't depend on actual checkpoint.
    """
    from services.converter import CT2PETConverter
    from utils.image_processing import ImageProcessor

    converter = CT2PETConverter.__new__(CT2PETConverter)
    converter.device = "cpu"
    converter.image_processor = ImageProcessor()
    converter.model = mock_model

    return converter
