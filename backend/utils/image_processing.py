import io
from typing import Any

from nibabel.loadsave import load, save
from nibabel.nifti1 import Nifti1Image
import numpy as np
from PIL import Image


class ImageProcessor:
    def load_nifti(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        # 读取 NIfTI 并返回 (volume, affine)。项目只支持 3D 体数据。
        nifti: Any = load(file_path)
        volume = np.asarray(nifti.get_fdata(), dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError("Only 3D NIfTI volumes are supported")
        return volume, nifti.affine

    def save_nifti(
        self, volume: np.ndarray, affine: np.ndarray, output_path: str
    ) -> None:
        nifti = Nifti1Image(volume.astype(np.float32), affine)
        save(nifti, output_path)

    def preprocess_ct_volume(
        self,
        volume: np.ndarray,
        min_hu: float = -900.0,
        max_hu: float = 200.0,
    ) -> np.ndarray:
        # CT 预处理：把 HU 裁剪到固定范围并归一化到 [0, 1]，与训练/推理期保持一致。
        clipped = np.clip(volume, min_hu, max_hu)
        normalized = (clipped - min_hu) / (max_hu - min_hu)
        return normalized.astype(np.float32)

    def edge_zero(self, stack: np.ndarray) -> np.ndarray:
        stack[:, 0, :] = 0
        stack[:, -1, :] = 0
        stack[:, :, 0] = 0
        stack[:, :, -1] = 0
        return stack

    def pad_volume_edge(self, volume: np.ndarray, pad_slices: int = 3) -> np.ndarray:
        return np.pad(volume, ((0, 0), (0, 0), (pad_slices, pad_slices)), mode="edge")

    def to_grayscale_png_bytes(self, slice_2d: np.ndarray) -> bytes:
        normalized = self._normalize(slice_2d)
        img = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
        return self._image_to_bytes(img)

    def to_hot_png_bytes(self, slice_2d: np.ndarray) -> bytes:
        normalized = self._normalize(slice_2d)
        rgb = self._hot_colormap(normalized)
        img = Image.fromarray(rgb, mode="RGB")
        return self._image_to_bytes(img)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if data_max - data_min < 1e-8:
            return np.zeros_like(data, dtype=np.float32)
        return (data - data_min) / (data_max - data_min)

    def _hot_colormap(self, values: np.ndarray) -> np.ndarray:
        x = np.clip(values, 0.0, 1.0)
        r = np.clip(3.0 * x, 0.0, 1.0)
        g = np.clip(3.0 * x - 1.0, 0.0, 1.0)
        b = np.clip(3.0 * x - 2.0, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
