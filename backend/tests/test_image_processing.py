"""
Unit tests for image processing helpers.

Tests cover:
- NIfTI load/save round-trip
- HU normalization clipping to [-900, 200]
- Edge padding behavior (3-slice replication)
- Invalid volume handling (empty, wrong dimensions)
"""

import numpy as np
import pytest

from utils.image_processing import ImageProcessor


class TestImageProcessorNiftiIO:
    """Test NIfTI file I/O operations."""

    def test_load_nifti_success(self, minimal_ct_volume):
        """Load a valid NIfTI file and verify shape/dtype."""
        filepath, original_volume, original_affine = minimal_ct_volume
        processor = ImageProcessor()

        volume, affine = processor.load_nifti(filepath)

        assert volume.shape == original_volume.shape
        assert volume.dtype == np.float32
        np.testing.assert_array_almost_equal(affine, original_affine)

    def test_load_nifti_invalid_2d_raises_error(self, invalid_2d_volume):
        """2D NIfTI should raise ValueError."""
        filepath, _, _ = invalid_2d_volume
        processor = ImageProcessor()

        with pytest.raises(ValueError, match="Only 3D NIfTI volumes are supported"):
            processor.load_nifti(filepath)

    def test_load_nifti_invalid_4d_raises_error(self, invalid_4d_volume):
        """4D NIfTI should raise ValueError."""
        filepath, _, _ = invalid_4d_volume
        processor = ImageProcessor()

        with pytest.raises(ValueError, match="Only 3D NIfTI volumes are supported"):
            processor.load_nifti(filepath)

    def test_load_nifti_corrupt_file_raises_error(self, corrupt_nifti_file):
        """Corrupt NIfTI file should raise ImageFileError or similar."""
        processor = ImageProcessor()

        with pytest.raises(
            Exception
        ):  # nibabel raises various errors for corrupt files
            processor.load_nifti(corrupt_nifti_file)

    def test_save_nifti_success(self, tmp_path, standard_ct_volume):
        """Save NIfTI volume and verify it can be loaded back."""
        _, original_volume, original_affine = standard_ct_volume
        output_path = str(tmp_path / "output.nii.gz")
        processor = ImageProcessor()

        processor.save_nifti(original_volume, original_affine, output_path)
        loaded_volume, loaded_affine = processor.load_nifti(output_path)

        np.testing.assert_array_almost_equal(loaded_volume, original_volume, decimal=4)
        np.testing.assert_array_almost_equal(loaded_affine, original_affine)

    def test_roundtrip_nifti_preserves_shape(self, tmp_path, minimal_ct_volume):
        """Load -> Save -> Load preserves shape and values."""
        filepath, _, _ = minimal_ct_volume
        processor = ImageProcessor()

        volume1, affine1 = processor.load_nifti(filepath)
        output_path = str(tmp_path / "roundtrip.nii.gz")
        processor.save_nifti(volume1, affine1, output_path)
        volume2, affine2 = processor.load_nifti(output_path)

        assert volume2.shape == volume1.shape
        np.testing.assert_array_almost_equal(volume2, volume1, decimal=4)


class TestImageProcessorNormalization:
    """Test CT HU normalization."""

    def test_preprocess_clips_to_hu_range(self):
        """Verify clipping to [-900, 200] HU range."""
        processor = ImageProcessor()

        # Create volume with values outside HU range
        volume = np.array([[-1000, -900, 0, 200, 500]], dtype=np.float32).reshape(
            1, 1, 5
        )
        normalized = processor.preprocess_ct_volume(volume, min_hu=-900.0, max_hu=200.0)

        # Clipped values should be in range
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Check specific clipped values
        # -1000 clips to -900 -> 0.0
        # -900 clips to -900 -> 0.0
        # 0 -> (0 - (-900)) / 1100 = 0.818...
        # 200 -> (200 - (-900)) / 1100 = 1.0
        # 500 clips to 200 -> 1.0
        expected = np.array(
            [[0.0, 0.0, 900.0 / 1100, 1.0, 1.0]], dtype=np.float32
        ).reshape(1, 1, 5)
        np.testing.assert_array_almost_equal(normalized, expected, decimal=2)

    def test_preprocess_default_hu_range(self):
        """Test default HU range [-900, 200]."""
        processor = ImageProcessor()

        # Values: -900 (min), 200 (max), 50 (mid)
        volume = np.array([[-900, 200, 50]], dtype=np.float32).reshape(1, 1, 3)
        normalized = processor.preprocess_ct_volume(volume)

        # After normalization to [0, 1]
        assert normalized[0, 0, 0] == pytest.approx(0.0)  # -900 -> 0
        assert normalized[0, 0, 1] == pytest.approx(1.0)  # 200 -> 1
        # 50: (50 - (-900)) / (200 - (-900)) = 950 / 1100 ≈ 0.8636
        assert normalized[0, 0, 2] == pytest.approx(0.8636, abs=0.01)

    def test_preprocess_output_dtype_float32(self):
        """Output should be float32."""
        processor = ImageProcessor()
        volume = np.random.rand(4, 4, 4).astype(np.float32)

        normalized = processor.preprocess_ct_volume(volume)

        assert normalized.dtype == np.float32

    def test_preprocess_all_zeros(self):
        """Test normalization of all-zero volume."""
        processor = ImageProcessor()
        volume = np.zeros((4, 4, 4), dtype=np.float32)

        normalized = processor.preprocess_ct_volume(volume)

        # All zeros -> normalized to [0, 1] range, middle value
        assert normalized.shape == (4, 4, 4)
        assert np.isfinite(normalized).all()


class TestImageProcessorEdgePadding:
    """Test edge padding behavior."""

    def test_pad_volume_edge_default_3_slices(self):
        """Default padding should be 3 slices on each side using edge replication."""
        processor = ImageProcessor()
        volume = np.ones((16, 16, 4), dtype=np.float32)

        padded = processor.pad_volume_edge(volume)

        # Original: (16, 16, 4)
        # Padded: (16, 16, 4+3+3) = (16, 16, 10)
        assert padded.shape == (16, 16, 10)

        # Check that edges replicate (edge mode replicates edge values)
        # First 3 should have same values as first slice (all ones)
        assert np.all(padded[:, :, 0:3] == 1.0)
        # Original slices should be in center (indices 3-6)
        np.testing.assert_array_equal(padded[:, :, 3:7], volume)
        # Last 3 should have same values as last slice (all ones)
        assert np.all(padded[:, :, 7:10] == 1.0)

    def test_pad_volume_edge_custom_pad_slices(self):
        """Test custom padding amount."""
        processor = ImageProcessor()
        volume = np.arange(16 * 16 * 4).reshape(16, 16, 4).astype(np.float32)

        padded = processor.pad_volume_edge(volume, pad_slices=2)

        assert padded.shape == (16, 16, 8)

        # First 2 should match first slice (edge replication)
        # All values should match the original first slice value
        for i in range(16):
            for j in range(16):
                assert padded[i, j, 0] == volume[i, j, 0]
                assert padded[i, j, 1] == volume[i, j, 0]
        # Center should have original volume (indices 2-5)
        np.testing.assert_array_equal(padded[:, :, 2:6], volume)
        # Last 2 should match last slice (edge replication)
        for i in range(16):
            for j in range(16):
                assert padded[i, j, 6] == volume[i, j, -1]
                assert padded[i, j, 7] == volume[i, j, -1]

    def test_pad_volume_preserves_center(self):
        """Padded volume should preserve original volume in center."""
        processor = ImageProcessor()
        volume = np.random.rand(16, 16, 4).astype(np.float32)

        padded = processor.pad_volume_edge(volume, pad_slices=3)

        # Center should match original (slices 3-7)
        np.testing.assert_array_equal(padded[:, :, 3:7], volume[:, :, :])


class TestImageProcessorEdgeZero:
    """Test edge zeroing for 7-slice stacks."""

    def test_edge_zero_sets_borders_to_zero(self):
        """Border voxels should be set to 0."""
        processor = ImageProcessor()
        stack = np.ones((7, 16, 16), dtype=np.float32)  # 7-slice stack, HxW

        result = processor.edge_zero(stack)

        # Check borders are zero
        assert (result[:, 0, :] == 0).all()  # Top edge
        assert (result[:, -1, :] == 0).all()  # Bottom edge
        assert (result[:, :, 0] == 0).all()  # Left edge
        assert (result[:, :, -1] == 0).all()  # Right edge

        # Center should remain 1
        assert (result[:, 1:-1, 1:-1] == 1).all()

    def test_edge_zero_preserves_center(self):
        """Interior voxels should be preserved."""
        processor = ImageProcessor()
        stack = np.arange(7 * 16 * 16).reshape(7, 16, 16).astype(np.float32)

        original_center = stack[:, 1:-1, 1:-1].copy()
        result = processor.edge_zero(stack)

        # Center should match
        np.testing.assert_array_equal(result[:, 1:-1, 1:-1], original_center)


class TestImageProcessorPngConversion:
    """Test PNG slice conversion."""

    def test_to_grayscale_png_bytes_returns_valid_png(self):
        """Grayscale conversion should return valid PNG bytes."""
        processor = ImageProcessor()
        slice_2d = np.random.rand(64, 64).astype(np.float32)

        png_bytes = processor.to_grayscale_png_bytes(slice_2d)

        # PNG magic bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png_bytes) > 0

    def test_to_hot_png_bytes_returns_valid_png(self):
        """Hot colormap conversion should return valid PNG bytes."""
        processor = ImageProcessor()
        slice_2d = np.random.rand(64, 64).astype(np.float32)

        png_bytes = processor.to_hot_png_bytes(slice_2d)

        # PNG magic bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(png_bytes) > 0

    def test_png_conversion_handles_uniform_slice(self):
        """Uniform slices (all same value) should not error."""
        processor = ImageProcessor()
        slice_2d = np.ones((64, 64), dtype=np.float32) * 0.5

        png_bytes_gray = processor.to_grayscale_png_bytes(slice_2d)
        png_bytes_hot = processor.to_hot_png_bytes(slice_2d)

        assert png_bytes_gray[:8] == b"\x89PNG\r\n\x1a\n"
        assert png_bytes_hot[:8] == b"\x89PNG\r\n\x1a\n"

    def test_png_conversion_handles_zero_slice(self):
        """All-zero slices should not error."""
        processor = ImageProcessor()
        slice_2d = np.zeros((64, 64), dtype=np.float32)

        png_bytes_gray = processor.to_grayscale_png_bytes(slice_2d)
        png_bytes_hot = processor.to_hot_png_bytes(slice_2d)

        assert png_bytes_gray[:8] == b"\x89PNG\r\n\x1a\n"
        assert png_bytes_hot[:8] == b"\x89PNG\r\n\x1a\n"
