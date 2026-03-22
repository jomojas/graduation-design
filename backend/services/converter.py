from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Literal, Optional

import numpy as np
import torch

from models.generator import load_model
from utils.file_utils import get_checkpoint_path
from utils.image_processing import ImageProcessor


logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    pred_pet_path: str
    num_slices: int
    shape: tuple[int, int, int]
    inference_status: Literal["pending", "processing", "completed", "failed"] = (
        "completed"
    )
    inference_started_at: Optional[datetime] = None
    inference_completed_at: Optional[datetime] = None
    inference_duration_ms: Optional[float] = None
    inference_error: Optional[str] = None


@dataclass
class InferenceEngineResult:
    status: Literal["pending", "processing", "completed", "failed"]
    num_slices: int
    shape: tuple[int, int, int]
    pred_volume: np.ndarray
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    error: Optional[str] = None


class SlidingWindowInferenceEngine:
    """Canonical 2.5D inference engine preserving existing model semantics."""

    def __init__(self, pad_slices: int = 3):
        self.pad_slices = pad_slices

    def run(
        self,
        *,
        ct_volume: np.ndarray,
        model: torch.nn.Module,
        device: str,
        image_processor: ImageProcessor,
    ) -> InferenceEngineResult:
        started_at = datetime.utcnow()
        if ct_volume.shape[2] < 1:
            raise ValueError("CT volume has no slices")

        padded = image_processor.pad_volume_edge(ct_volume, pad_slices=self.pad_slices)
        pred_volume = np.zeros(ct_volume.shape, dtype=np.float32)
        with torch.no_grad():
            for i in range(self.pad_slices, padded.shape[2] - self.pad_slices):
                stack = padded[:, :, i - self.pad_slices : i + self.pad_slices + 1]
                stack = np.transpose(stack, (2, 0, 1))
                stack = image_processor.edge_zero(stack)
                input_tensor = torch.from_numpy(stack).unsqueeze(0).to(device)
                output = model(input_tensor)
                output_np = output.squeeze(0).detach().cpu().numpy()
                pred_volume[:, :, i - self.pad_slices] = output_np[1]

        completed_at = datetime.utcnow()
        duration_ms = (completed_at - started_at).total_seconds() * 1000.0
        volume_shape: tuple[int, int, int] = (
            int(ct_volume.shape[0]),
            int(ct_volume.shape[1]),
            int(ct_volume.shape[2]),
        )
        return InferenceEngineResult(
            status="completed",
            num_slices=int(ct_volume.shape[2]),
            shape=volume_shape,
            pred_volume=pred_volume,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=None,
        )


class CT2PETConverter:
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_processor = ImageProcessor()

        if model_path is None:
            model_path = get_checkpoint_path("generator.pth")

        self.model = load_model(model_path, self.device)
        self.engine = SlidingWindowInferenceEngine(pad_slices=3)
        self.model.eval()
        logger.info("Model loaded on %s", self.device)

    def convert_nifti(self, ct_path: str, output_path: str) -> ConversionResult:
        ct_volume, affine = self.image_processor.load_nifti(ct_path)
        ct_volume = self.image_processor.preprocess_ct_volume(ct_volume)

        engine = getattr(self, "engine", SlidingWindowInferenceEngine(pad_slices=3))
        self.engine = engine
        engine_result = engine.run(
            ct_volume=ct_volume,
            model=self.model,
            device=self.device,
            image_processor=self.image_processor,
        )

        self.image_processor.save_nifti(engine_result.pred_volume, affine, output_path)
        return ConversionResult(
            pred_pet_path=output_path,
            num_slices=engine_result.num_slices,
            shape=engine_result.shape,
            inference_status=engine_result.status,
            inference_started_at=engine_result.started_at,
            inference_completed_at=engine_result.completed_at,
            inference_duration_ms=engine_result.duration_ms,
            inference_error=engine_result.error,
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
