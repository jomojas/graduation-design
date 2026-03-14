import os
import torch
from PIL import Image

from models.generator import Generator, load_model
from utils.image_processing import ImageProcessor
from utils.file_utils import get_output_path, get_checkpoint_path


class CT2PETConverter:
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_processor = ImageProcessor(target_size=(256, 256))
        
        if model_path is None:
            model_path = get_checkpoint_path("generator.pth")
        
        self.model = load_model(model_path, self.device)
        self.model.eval()
        
        print(f"模型已加载到 {self.device}")

    def convert(self, ct_image: Image.Image) -> Image.Image:
        input_tensor = self.image_processor.preprocess(ct_image)
        
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_batch)
        
        output_image = self.image_processor.postprocess(output_tensor)
        
        return output_image

    def convert_bytes(self, image_bytes: bytes) -> bytes:
        ct_image = self.image_processor.load_image(image_bytes)
        
        pet_image = self.convert(ct_image)
        
        output_path = get_output_path()
        self.image_processor.save_image(pet_image, output_path)
        
        with open(output_path, 'rb') as f:
            result_bytes = f.read()
        
        os.remove(output_path)
        
        return result_bytes

    def convert_and_save(self, ct_image: Image.Image) -> str:
        pet_image = self.convert(ct_image)
        
        output_path = get_output_path()
        self.image_processor.save_image(pet_image, output_path)
        
        return output_path


converter_instance = None


def get_converter(model_path: str = None) -> CT2PETConverter:
    global converter_instance
    if converter_instance is None:
        converter_instance = CT2PETConverter(model_path)
    return converter_instance
