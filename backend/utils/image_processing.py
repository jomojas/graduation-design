import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class ImageProcessor:
    def __init__(self, target_size: tuple = (256, 256)):
        self.target_size = target_size
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1], std=[2]),
            transforms.ToPILImage()
        ])

    def load_image(self, image_data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB' and image.mode != 'L':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"无法加载图像: {e}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        tensor = tensor.cpu().detach()
        
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        
        image = self.inverse_transform(tensor)
        
        return image

    def preprocess_batch(self, images: list) -> torch.Tensor:
        tensors = [self.preprocess(img) for img in images]
        return torch.stack(tensors)

    def save_image(self, image: Image.Image, output_path: str):
        image.save(output_path, format='PNG')

    def image_to_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.getvalue()
