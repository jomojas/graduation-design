from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from models.generator import load_model
from utils.file_utils import get_checkpoint_path
from utils.image_processing import ImageProcessor


@dataclass
class ConversionResult:
    pred_pet_path: str
    num_slices: int
    shape: tuple[int, int, int]


class CT2PETConverter:
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_processor = ImageProcessor()

        if model_path is None:
            model_path = get_checkpoint_path("generator.pth")

        self.model = load_model(model_path, self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def convert_nifti(self, ct_path: str, output_path: str) -> ConversionResult:
        ct_volume, affine = self.image_processor.load_nifti(ct_path)
        ct_volume = self.image_processor.preprocess_ct_volume(ct_volume)

        if ct_volume.shape[2] < 1:
            raise ValueError("CT volume has no slices")

        padded = self.image_processor.pad_volume_edge(ct_volume, pad_slices=3)

        pred_volume = np.zeros(ct_volume.shape, dtype=np.float32)
        with torch.no_grad():
            for i in range(3, padded.shape[2] - 3):
                stack = padded[:, :, i - 3 : i + 4]
                stack = np.transpose(stack, (2, 0, 1))
                stack = self.image_processor.edge_zero(stack)
                input_tensor = torch.from_numpy(stack).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                output_np = output.squeeze(0).detach().cpu().numpy()
                pred_volume[:, :, i - 3] = output_np[1]

        self.image_processor.save_nifti(pred_volume, affine, output_path)
        volume_shape: tuple[int, int, int] = (
            int(ct_volume.shape[0]),
            int(ct_volume.shape[1]),
            int(ct_volume.shape[2]),
        )
        return ConversionResult(
            pred_pet_path=output_path,
            num_slices=int(ct_volume.shape[2]),
            shape=volume_shape,
        )

    def get_slice_png(
        self,
        volume_path: str,
        index: int,
        view: Literal["ct", "real_pet", "pred_pet"],
    ) -> bytes:
        volume, _ = self.image_processor.load_nifti(volume_path)
        if index < 0 or index >= volume.shape[2]:
            raise IndexError("Slice index out of range")

        slice_2d = volume[:, :, index]
        if view == "ct":
            return self.image_processor.to_grayscale_png_bytes(slice_2d)
        return self.image_processor.to_hot_png_bytes(slice_2d)


converter_instance: Optional[CT2PETConverter] = None


def get_converter(model_path: Optional[str] = None) -> CT2PETConverter:
    global converter_instance
    if converter_instance is None:
        converter_instance = CT2PETConverter(model_path)
    return converter_instance
