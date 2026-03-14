import os
import uuid
import shutil
from pathlib import Path


UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"


def ensure_directories():
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def generate_unique_filename(extension: str = "png") -> str:
    return f"{uuid.uuid4().hex}.{extension}"


def get_upload_path(filename: str = None) -> str:
    if filename is None:
        filename = generate_unique_filename()
    return os.path.join(UPLOAD_DIR, filename)


def get_output_path(filename: str = None) -> str:
    if filename is None:
        filename = generate_unique_filename()
    return os.path.join(OUTPUT_DIR, filename)


def get_checkpoint_path(model_name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, model_name)


def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    import time
    current_time = time.time()
    
    if not os.path.exists(directory):
        return
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                except Exception:
                    pass


def save_upload_file(file_content: bytes, filename: str = None) -> str:
    if filename is None:
        filename = generate_unique_filename()
    
    filepath = get_upload_path(filename)
    with open(filepath, 'wb') as f:
        f.write(file_content)
    
    return filepath


def read_file(filepath: str) -> bytes:
    with open(filepath, 'rb') as f:
        return f.read()
