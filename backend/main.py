import os
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from services.converter import get_converter
from utils.file_utils import ensure_directories, get_checkpoint_path


model_loaded = False
converter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded, converter

    ensure_directories()

    model_path = get_checkpoint_path("generator.pth")

    if os.path.exists(model_path):
        try:
            converter = get_converter(model_path)
            model_loaded = True
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            model_loaded = False
    else:
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用随机初始化的模型进行测试")
        model_loaded = False

    yield

    print("应用关闭")


app = FastAPI(
    title="CT to PET Image Converter",
    description="基于深度学习的CT转PET图像生成服务",
    version="1.0.0",
    lifespan=lifespan,
)


class ConvertResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    pet_image_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@app.get("/", response_model=StatusResponse)
async def root():
    device = (
        "cuda"
        if converter and hasattr(converter, "device") and converter.device == "cuda"
        else "cpu"
    )
    return StatusResponse(status="ok", model_loaded=model_loaded, device=device)


@app.get("/status")
async def get_status():
    device = (
        "cuda"
        if converter and hasattr(converter, "device") and converter.device == "cuda"
        else "cpu"
    )
    return {"status": "running", "model_loaded": model_loaded, "device": device}


@app.post("/convert", response_model=ConvertResponse)
async def convert_ct_to_pet(file: UploadFile = File(...)):
    global converter
    if not file:
        raise HTTPException(status_code=400, detail="未上传文件")

    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, detail="不支持的文件类型，请上传JPG或PNG格式"
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小超过10MB限制")

        if converter is None:
            from services.converter import CT2PETConverter

            converter = CT2PETConverter()

        pet_image_bytes = converter.convert_bytes(image_bytes)

        base64_image = base64.b64encode(pet_image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"

        return ConvertResponse(success=True, message="转换成功", pet_image_url=data_url)

    except HTTPException:
        raise
    except Exception as e:
        print(f"转换错误: {e}")
        return ConvertResponse(
            success=False, message=f"转换失败: {str(e)}", pet_image_url=None
        )
