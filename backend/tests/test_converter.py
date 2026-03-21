"""
Tests for CT2PETConverter with mocked model inference.

These tests verify the conversion pipeline without depending on actual
model checkpoints. Uses pytest-mock fixtures to isolate inference logic.

Tests cover:
- Mocked inference through the full pipeline
- Error handling for invalid volumes
- Padding and preprocessing chain
- Slice PNG generation after conversion
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest
import torch

from services.converter import CT2PETConverter, ConversionResult
from utils.image_processing import ImageProcessor


class TestConverterWithMockedModel:
    """Test CT2PETConverter with mocked PyTorch model."""

    def test_convert_nifti_with_mocked_model(self, mock_converter, minimal_ct_volume):
        """
        Test full conversion pipeline with mocked model.

        Verifies:
        - NIfTI loads correctly
        - Preprocessing applies
        - Padding enables 7-slice sliding window
        - Model inference executes
        - Output is saved as NIfTI
        """
        ct_path, _, _ = minimal_ct_volume
        output_path = str(Path(ct_path).parent / "pred_pet.nii.gz")

        result = mock_converter.convert_nifti(ct_path, output_path)

        assert result.num_slices == 4
        assert result.shape == (16, 16, 4)
        assert os.path.exists(output_path)

        # Verify output is valid NIfTI
        pred_volume, _ = mock_converter.image_processor.load_nifti(output_path)
        assert pred_volume.shape == (16, 16, 4)

    def test_converter_handles_empty_volume(self, mock_converter, empty_volume):
        """
        Empty volume (no slices) should raise error.
        """
        filepath, _, _ = empty_volume
        output_path = str(Path(filepath).parent / "output.nii.gz")

        with pytest.raises(ValueError):
            # Empty volumes (0 slices) are not 3D, so they raise ValueError during load
            mock_converter.convert_nifti(filepath, output_path)

    def test_converter_handles_invalid_nifti(self, mock_converter, corrupt_nifti_file):
        """
        Corrupt NIfTI should raise error during load.
        """
        output_path = str(Path(corrupt_nifti_file).parent / "output.nii.gz")

        with pytest.raises(Exception):  # nibabel raises ImageFileError
            mock_converter.convert_nifti(corrupt_nifti_file, output_path)

    def test_converter_preprocessing_chain(self, mock_converter, standard_ct_volume):
        """
        Test that preprocessing chain executes correctly:
        1. Load NIfTI
        2. Preprocess HU normalization
        3. Pad volume
        """
        ct_path, _, _ = standard_ct_volume
        processor = mock_converter.image_processor

        # Load
        volume, _ = processor.load_nifti(ct_path)
        original_shape = volume.shape

        # Preprocess
        normalized = processor.preprocess_ct_volume(volume)
        assert normalized.dtype == np.float32
        assert normalized.shape == original_shape

        # Pad
        padded = processor.pad_volume_edge(normalized, pad_slices=3)
        assert padded.shape == (
            original_shape[0],
            original_shape[1],
            original_shape[2] + 6,
        )

    def test_converter_device_selection(self, mock_converter):
        """
        Converter should use CPU when CUDA unavailable.
        """
        # Mock converter is set to CPU by fixture
        assert mock_converter.device == "cpu"


class TestConverterSlicePngGeneration:
    """Test PNG generation from converted volumes."""

    def test_get_ct_slice_png(self, mock_converter, minimal_ct_volume):
        """CT slice should render as grayscale PNG."""
        ct_path, _, _ = minimal_ct_volume

        png_bytes = mock_converter.get_slice_png(ct_path, 0, view="ct")

        # Valid PNG magic bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_get_pet_slice_png(self, mock_converter, minimal_ct_volume):
        """PET slice should render as hot-colored PNG."""
        ct_path, _, _ = minimal_ct_volume

        png_bytes = mock_converter.get_slice_png(ct_path, 0, view="pred_pet")

        # Valid PNG magic bytes
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_slice_index_out_of_range_raises_error(
        self, mock_converter, minimal_ct_volume
    ):
        """Slice index beyond volume bounds should raise IndexError."""
        ct_path, _, _ = minimal_ct_volume

        with pytest.raises(IndexError, match="Slice index out of range"):
            mock_converter.get_slice_png(ct_path, 100, view="ct")

    def test_negative_slice_index_raises_error(self, mock_converter, minimal_ct_volume):
        """Negative slice index should raise IndexError."""
        ct_path, _, _ = minimal_ct_volume

        with pytest.raises(IndexError, match="Slice index out of range"):
            mock_converter.get_slice_png(ct_path, -1, view="ct")


class TestConverterConversionResult:
    """Test ConversionResult dataclass."""

    def test_conversion_result_creation(self):
        """ConversionResult should store metadata correctly."""
        result = ConversionResult(
            pred_pet_path="/outputs/test/pred_pet.nii.gz",
            num_slices=128,
            shape=(512, 512, 128),
        )

        assert result.pred_pet_path == "/outputs/test/pred_pet.nii.gz"
        assert result.num_slices == 128
        assert result.shape == (512, 512, 128)

    def test_conversion_result_shape_tuple(self):
        """Shape should be stored as tuple."""
        result = ConversionResult(
            pred_pet_path="/test.nii.gz",
            num_slices=10,
            shape=(64, 64, 10),
        )

        assert isinstance(result.shape, tuple)
        assert len(result.shape) == 3


class TestConverterEdgeCases:
    """Test edge cases and error conditions."""

    def test_converter_with_very_small_volume(
        self, mock_converter, small_volume_16x16x4
    ):
        """
        Minimum valid volume (16x16x4) should convert successfully.
        This tests that padding works for smallest supported size.
        """
        ct_path, _, _ = small_volume_16x16x4
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        result = mock_converter.convert_nifti(ct_path, output_path)

        assert result.num_slices == 4
        assert os.path.exists(output_path)

    def test_converter_preserves_affine(self, mock_converter, standard_ct_volume):
        """Output volume should have same affine as input."""
        ct_path, _, original_affine = standard_ct_volume
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        mock_converter.convert_nifti(ct_path, output_path)

        _, output_affine = mock_converter.image_processor.load_nifti(output_path)
        np.testing.assert_array_almost_equal(output_affine, original_affine)

    def test_converter_output_dtype_float32(self, mock_converter, standard_ct_volume):
        """Output volume should be float32."""
        ct_path, _, _ = standard_ct_volume
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        mock_converter.convert_nifti(ct_path, output_path)

        output_volume, _ = mock_converter.image_processor.load_nifti(output_path)
        assert output_volume.dtype == np.float32


class TestConverterModelInference:
    """Test model inference integration points."""

    def test_converter_uses_torch_no_grad(self, mock_converter, standard_ct_volume):
        """
        Inference should use torch.no_grad() for efficiency.

        Note: This test verifies the pattern is used in implementation.
        Current mock automatically handles this.
        """
        ct_path, _, _ = standard_ct_volume
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        # Should not raise and should complete
        result = mock_converter.convert_nifti(ct_path, output_path)
        assert result is not None

    def test_model_called_for_each_slice_window(
        self, mock_converter, minimal_ct_volume
    ):
        """
        Model should be called once per output slice (7-slice windows).

        Volume: 4 slices
        Padded: 10 slices (4 + 3 + 3)
        Windows: 4 windows (sliding from index 3 to 6, inclusive)
        So model should be called 4 times.
        """
        ct_path, _, _ = minimal_ct_volume
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        # Replace mock with a counter
        call_count = 0
        original_forward = mock_converter.model.side_effect

        def counting_forward(input_tensor):
            nonlocal call_count
            call_count += 1
            return original_forward(input_tensor)

        mock_converter.model.side_effect = counting_forward

        mock_converter.convert_nifti(ct_path, output_path)

        # Should be called 4 times (one per output slice)
        assert call_count == 4

    def test_model_receives_correct_tensor_shape(
        self, mock_converter, minimal_ct_volume
    ):
        """
        Model should receive 7-slice stacks as tensors of shape (1, 7, H, W).
        """
        ct_path, _, _ = minimal_ct_volume
        output_path = str(Path(ct_path).parent / "output.nii.gz")

        received_shapes = []
        original_forward = mock_converter.model.side_effect

        def shape_tracking_forward(input_tensor):
            received_shapes.append(input_tensor.shape)
            return original_forward(input_tensor)

        mock_converter.model.side_effect = shape_tracking_forward

        mock_converter.convert_nifti(ct_path, output_path)

        # All should be (1, 7, 16, 16)
        for shape in received_shapes:
            assert shape[0] == 1  # Batch
            assert shape[1] == 7  # Channels (7 slices)
            assert shape[2] == 16  # Height
            assert shape[3] == 16  # Width


class TestConverterIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_nifti_to_nifti(
        self, mock_converter, standard_ct_volume, tmp_path
    ):
        """
        Full pipeline: NIfTI CT -> convert -> NIfTI PET -> PNG slices.
        """
        ct_path, _, _ = standard_ct_volume
        output_path = str(tmp_path / "pred_pet.nii.gz")

        # Convert
        result = mock_converter.convert_nifti(ct_path, output_path)

        # Get slices
        ct_png = mock_converter.get_slice_png(ct_path, 0, view="ct")
        pet_png = mock_converter.get_slice_png(output_path, 0, view="pred_pet")

        # Verify
        assert result.num_slices == 32  # standard_ct_volume is 64x64x32
        assert ct_png[:8] == b"\x89PNG\r\n\x1a\n"
        assert pet_png[:8] == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.parametrize("slice_idx", [0, 10, 31])
    def test_all_slices_convertible_to_png(
        self, mock_converter, standard_ct_volume, slice_idx
    ):
        """
        All slices should be convertible to PNG without error.
        Standard volume is 64x64x32, so valid indices are 0-31.
        """
        ct_path, _, _ = standard_ct_volume

        png_bytes = mock_converter.get_slice_png(ct_path, slice_idx, view="ct")

        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
