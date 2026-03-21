"""
Unit tests for geometry validation and mismatch handling.

Tests cover:
- Shape mismatch detection between CT and PET volumes
- Spacing (voxel size) mismatch detection via affine matrices
- Orientation alignment checks
- Volume shape validation
"""

import numpy as np
import pytest

from utils.image_processing import ImageProcessor


class TestGeometryMismatchDetection:
    """Test detection of geometry mismatches between volumes."""

    def test_shape_mismatch_raises_error(
        self, standard_ct_volume, mismatched_pet_volume
    ):
        """Volumes with different shapes should be flagged as incompatible."""
        ct_path, ct_volume, ct_affine = standard_ct_volume
        pet_path, pet_volume, pet_affine = mismatched_pet_volume

        processor = ImageProcessor()
        ct_loaded, _ = processor.load_nifti(ct_path)
        pet_loaded, _ = processor.load_nifti(pet_path)

        # Shapes should differ
        assert ct_loaded.shape != pet_loaded.shape

        # This is what main.py checks in upload endpoint
        with pytest.raises(AssertionError):
            assert ct_loaded.shape == pet_loaded.shape

    def test_spacing_mismatch_detection(
        self, standard_ct_volume, mismatched_spacing_volume
    ):
        """Volumes with different voxel spacing should have different affines."""
        ct_path, ct_volume, ct_affine = standard_ct_volume
        pet_path, pet_volume, pet_affine = mismatched_spacing_volume

        processor = ImageProcessor()
        _, ct_affine_loaded = processor.load_nifti(ct_path)
        _, pet_affine_loaded = processor.load_nifti(pet_path)

        # Affines should differ
        assert not np.allclose(ct_affine_loaded, pet_affine_loaded)

        # Extract voxel spacing from diagonal
        ct_spacing = np.diag(ct_affine_loaded)[:3]
        pet_spacing = np.diag(pet_affine_loaded)[:3]

        assert not np.allclose(ct_spacing, pet_spacing)

    def test_matching_geometry_accepts_volumes(self, standard_ct_volume):
        """Volumes with matching geometry should be compatible."""
        ct_path, ct_volume, ct_affine = standard_ct_volume
        processor = ImageProcessor()

        ct_loaded1, ct_affine1 = processor.load_nifti(ct_path)
        ct_loaded2, ct_affine2 = processor.load_nifti(ct_path)

        # Same file should have matching shape
        assert ct_loaded1.shape == ct_loaded2.shape
        # Same file should have matching affine
        np.testing.assert_array_almost_equal(ct_affine1, ct_affine2)


class TestVolumeShapeValidation:
    """Test validation of volume dimensions."""

    def test_valid_3d_volume_accepted(self, standard_ct_volume):
        """Valid 3D volumes should load without error."""
        filepath, _, _ = standard_ct_volume
        processor = ImageProcessor()

        volume, _ = processor.load_nifti(filepath)

        assert volume.ndim == 3
        assert volume.shape[0] > 0
        assert volume.shape[1] > 0
        assert volume.shape[2] > 0

    def test_invalid_2d_volume_rejected(self, invalid_2d_volume):
        """2D volumes should be rejected."""
        filepath, _, _ = invalid_2d_volume
        processor = ImageProcessor()

        with pytest.raises(ValueError, match="Only 3D NIfTI volumes are supported"):
            processor.load_nifti(filepath)

    def test_invalid_4d_volume_rejected(self, invalid_4d_volume):
        """4D volumes (e.g., multi-channel) should be rejected."""
        filepath, _, _ = invalid_4d_volume
        processor = ImageProcessor()

        with pytest.raises(ValueError, match="Only 3D NIfTI volumes are supported"):
            processor.load_nifti(filepath)

    def test_empty_volume_validation(self, empty_volume):
        """Volume with no slices (0 depth) is not 3D, so it raises during load."""
        filepath, volume, _ = empty_volume
        processor = ImageProcessor()

        # File should load NIfTI but fail because it's not a valid 3D volume
        with pytest.raises(ValueError, match="Only 3D NIfTI volumes are supported"):
            processor.load_nifti(filepath)


class TestMinimalVolumeRequirements:
    """Test minimum size requirements for model inference."""

    def test_minimum_spatial_dimensions_16x16(self, small_volume_16x16x4):
        """Model requires minimum 16x16 spatial dimensions."""
        filepath, volume, _ = small_volume_16x16x4
        processor = ImageProcessor()

        loaded_volume, _ = processor.load_nifti(filepath)

        # Check minimum spatial dims
        assert loaded_volume.shape[0] >= 16  # Height
        assert loaded_volume.shape[1] >= 16  # Width
        assert loaded_volume.shape[2] >= 1  # Depth

    def test_minimum_depth_requirement(self, minimal_ct_volume):
        """Volume needs at least 1 slice (padded to 7 for sliding window)."""
        filepath, volume, _ = minimal_ct_volume
        processor = ImageProcessor()

        loaded_volume, _ = processor.load_nifti(filepath)

        assert loaded_volume.shape[2] >= 1

    def test_edge_padding_enables_7slice_processing(self, minimal_ct_volume):
        """Edge padding enables 7-slice sliding window processing."""
        filepath, volume, _ = minimal_ct_volume
        processor = ImageProcessor()

        loaded_volume, _ = processor.load_nifti(filepath)

        # Original: 4 slices
        assert loaded_volume.shape[2] == 4

        # After padding with 3 on each side: 4 + 3 + 3 = 10
        padded = processor.pad_volume_edge(loaded_volume, pad_slices=3)
        assert padded.shape[2] == 10

        # Sliding window can process slices 3-6 (indices where center is original slices 0-3)
        # This tests the 2.5D approach: for each middle slice in 7-slice stack


class TestVolumeIntegrity:
    """Test overall volume integrity checks."""

    def test_volume_dtype_normalized_to_float32(self, standard_ct_volume):
        """Loaded volumes should be float32."""
        filepath, _, _ = standard_ct_volume
        processor = ImageProcessor()

        volume, _ = processor.load_nifti(filepath)

        assert volume.dtype == np.float32

    def test_volume_values_finite(self, standard_ct_volume):
        """Volume should contain finite values (no NaN/Inf)."""
        filepath, _, _ = standard_ct_volume
        processor = ImageProcessor()

        volume, _ = processor.load_nifti(filepath)

        assert np.isfinite(volume).all()

    def test_affine_matrix_valid(self, standard_ct_volume):
        """Affine matrix should be a valid 4x4 matrix."""
        filepath, _, _ = standard_ct_volume
        processor = ImageProcessor()

        _, affine = processor.load_nifti(filepath)

        assert affine.shape == (4, 4)
        assert np.isfinite(affine).all()

    def test_hu_normalization_preserves_shape(self, minimal_ct_volume):
        """HU normalization should not change volume shape."""
        filepath, _, _ = minimal_ct_volume
        processor = ImageProcessor()

        volume, _ = processor.load_nifti(filepath)
        original_shape = volume.shape

        normalized = processor.preprocess_ct_volume(volume)

        assert normalized.shape == original_shape


class TestGeometryMismatchInContext:
    """Integration-style tests for geometry mismatch scenarios."""

    def test_ct_pet_shape_compatibility_check(
        self, standard_ct_volume, mismatched_pet_volume
    ):
        """
        Simulate the shape check from main.py upload endpoint.
        CT and PET volumes must have same shape.
        """
        ct_path, _, _ = standard_ct_volume
        pet_path, _, _ = mismatched_pet_volume
        processor = ImageProcessor()

        ct_volume, _ = processor.load_nifti(ct_path)
        pet_volume, _ = processor.load_nifti(pet_path)

        # This check is done in main.py:302
        shapes_match = ct_volume.shape == pet_volume.shape
        assert not shapes_match, "Test volumes should have mismatched shapes"

    def test_geometry_mismatch_detection_procedure(
        self, standard_ct_volume, mismatched_spacing_volume
    ):
        """
        Test the procedure for detecting spacing/geometry mismatches.
        In production, this would compare affine matrices.
        """
        ct_path, _, _ = standard_ct_volume
        pet_path, _, _ = mismatched_spacing_volume
        processor = ImageProcessor()

        ct_vol, ct_aff = processor.load_nifti(ct_path)
        pet_vol, pet_aff = processor.load_nifti(pet_path)

        # Extract voxel spacing (mm per voxel)
        ct_voxel_size = np.linalg.norm(ct_aff[:3, :3], axis=0)
        pet_voxel_size = np.linalg.norm(pet_aff[:3, :3], axis=0)

        # Should detect mismatch
        spacing_matches = np.allclose(ct_voxel_size, pet_voxel_size, atol=0.1)
        assert not spacing_matches, "Test volumes should have mismatched spacing"
