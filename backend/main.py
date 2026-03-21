import os
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from nibabel.filebasedimages import ImageFileError
from pydantic import BaseModel, ConfigDict, Field

from services.converter import CT2PETConverter, get_converter
from utils.file_utils import CHECKPOINT_DIR, OUTPUT_DIR, UPLOAD_DIR, ensure_directories


class InferenceStatus(BaseModel):
    """Track the status of inference execution."""

    state: Literal["pending", "processing", "completed", "failed"]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class MetricsState(BaseModel):
    """Track metrics collected during conversion."""

    inference_time_ms: Optional[float] = None
    output_shape: Optional[tuple[int, int, int]] = None
    slices_processed: Optional[int] = None


class StudyManifest(BaseModel):
    """
    Canonical study/job manifest covering all metadata and paths.

    Supports multiple source types (NIfTI, DICOM) and tracks the complete
    lifecycle of a case from upload through inference and optional evaluation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "abc123def456",
                "source_type": "nifti",
                "upload_mode": "with_evaluation",
                "patient_id": "P001",
                "patient_name": "John Doe",
                "study_id": "S001",
                "study_date": "2025-03-21",
                "ct_volume_path": "/uploads/abc123def456/ct.nii.gz",
                "real_pet_volume_path": "/uploads/abc123def456/real_pet.nii.gz",
                "pred_pet_volume_path": "/outputs/abc123def456/pred_pet.nii.gz",
                "num_slices": 128,
                "shape": [512, 512, 128],
                "inference_status": {
                    "state": "completed",
                    "started_at": "2025-03-21T10:00:00",
                    "completed_at": "2025-03-21T10:05:00",
                    "error": None,
                },
                "metrics": {
                    "inference_time_ms": 300000.0,
                    "output_shape": [512, 512, 128],
                    "slices_processed": 128,
                },
                "error_status": None,
                "created_at": "2025-03-21T10:00:00",
                "updated_at": "2025-03-21T10:05:00",
            }
        }
    )

    # Core identifiers
    job_id: str = Field(..., description="Unique job identifier (UUID hex)")
    source_type: Literal["nifti", "dicom_zip", "dicom_dir"] = Field(
        default="nifti", description="Type of input source"
    )
    upload_mode: Literal["inference_only", "with_evaluation"] = Field(
        default="inference_only",
        description="Whether evaluation data (real PET) is included",
    )

    # Patient/Study metadata
    patient_id: Optional[str] = Field(
        default=None, description="Patient identifier (if available)"
    )
    patient_name: Optional[str] = Field(
        default=None, description="Patient name (if available)"
    )
    study_id: Optional[str] = Field(
        default=None, description="Study identifier (if available)"
    )
    study_date: Optional[str] = Field(
        default=None, description="Study date in ISO format (if available)"
    )

    # Standardized volume paths
    ct_volume_path: str = Field(
        ..., description="Path to input CT volume (NIfTI format)"
    )
    real_pet_volume_path: Optional[str] = Field(
        default=None,
        description="Path to real PET volume (NIfTI format) for evaluation",
    )
    pred_pet_volume_path: str = Field(
        ..., description="Path to predicted PET volume output"
    )

    # Volume metadata
    num_slices: int = Field(..., description="Number of slices in volume")
    shape: tuple[int, int, int] = Field(
        ..., description="3D shape (height, width, depth) in voxels"
    )

    # Inference status tracking
    inference_status: InferenceStatus = Field(
        default_factory=lambda: InferenceStatus(state="pending"),
        description="Current inference execution status",
    )

    # Metrics collected post-inference
    metrics: MetricsState = Field(
        default_factory=MetricsState, description="Inference and output metrics"
    )

    # Error tracking
    error_status: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


@dataclass
class CaseRecord:
    """In-memory record for backward compatibility with NIfTI flow."""

    job_id: str
    ct_path: str
    pred_pet_path: str
    real_pet_path: Optional[str]
    num_slices: int
    shape: tuple[int, int, int]


model_loaded = False
converter: Optional[CT2PETConverter] = None
case_records: dict[str, CaseRecord] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded, converter
    ensure_directories()

    model_path = os.path.join(CHECKPOINT_DIR, "generator.pth")
    if os.path.exists(model_path):
        try:
            converter = get_converter(model_path)
            model_loaded = True
            print(f"Model loaded: {model_path}")
        except Exception as exc:
            print(f"Model load failed: {exc}")
            model_loaded = False
    else:
        print(f"Warning: checkpoint not found at {model_path}")
        model_loaded = False

    yield


app = FastAPI(
    title="CT to PET 2.5D Service",
    description="NIfTI CT to synthetic PET inference with optional evaluation mode",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class UploadResponse(BaseModel):
    success: bool
    job_id: str
    num_slices: int
    shape: tuple[int, int, int]
    has_real_pet: bool


class CaseMetaResponse(BaseModel):
    success: bool
    job_id: str
    num_slices: int
    shape: tuple[int, int, int]
    has_real_pet: bool


def _validate_nifti(filename: str) -> None:
    lower = filename.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise HTTPException(
            status_code=400, detail="Only .nii or .nii.gz files are supported"
        )


def _get_nifti_extension(content: bytes) -> str:
    """Detect NIfTI extension based on gzip magic bytes.

    Args:
        content: Raw file bytes

    Returns:
        ".nii.gz" if gzip-compressed, ".nii" otherwise
    """
    if content[:2] == b"\x1f\x8b":
        return ".nii.gz"
    return ".nii"


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    device = converter.device if converter else "cpu"
    return StatusResponse(status="ok", model_loaded=model_loaded, device=device)


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    device = converter.device if converter else "cpu"
    return StatusResponse(status="running", model_loaded=model_loaded, device=device)


@app.post("/upload", response_model=UploadResponse)
async def upload_case(
    ct_file: UploadFile = File(...),
    real_pet_file: Optional[UploadFile] = File(None),
) -> UploadResponse:
    global converter

    if converter is None:
        converter = get_converter(os.path.join(CHECKPOINT_DIR, "generator.pth"))

    if not ct_file.filename:
        raise HTTPException(status_code=400, detail="CT file is required")
    _validate_nifti(ct_file.filename)
    if real_pet_file and real_pet_file.filename:
        _validate_nifti(real_pet_file.filename)

    job_id = uuid.uuid4().hex
    upload_dir = Path(UPLOAD_DIR) / job_id
    output_dir = Path(OUTPUT_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_content = await ct_file.read()
    if len(ct_content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="CT file exceeds 200MB")
    ct_ext = _get_nifti_extension(ct_content)
    ct_path = upload_dir / f"ct{ct_ext}"
    ct_path.write_bytes(ct_content)

    real_pet_path: Optional[Path] = None
    if real_pet_file:
        pet_content = await real_pet_file.read()
        if len(pet_content) > 200 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Real PET file exceeds 200MB")
        pet_ext = _get_nifti_extension(pet_content)
        real_pet_path = upload_dir / f"real_pet{pet_ext}"
        real_pet_path.write_bytes(pet_content)

        try:
            ct_volume, _ = converter.image_processor.load_nifti(str(ct_path))
            real_pet_volume, _ = converter.image_processor.load_nifti(
                str(real_pet_path)
            )
        except (ImageFileError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {str(e)}")
        if ct_volume.shape != real_pet_volume.shape:
            raise HTTPException(
                status_code=400, detail="CT and Real PET volume shapes must match"
            )

    pred_pet_path = output_dir / "pred_pet.nii.gz"
    try:
        result = converter.convert_nifti(str(ct_path), str(pred_pet_path))
    except (ImageFileError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {str(e)}")

    case_records[job_id] = CaseRecord(
        job_id=job_id,
        ct_path=str(ct_path),
        pred_pet_path=str(pred_pet_path),
        real_pet_path=str(real_pet_path) if real_pet_path else None,
        num_slices=result.num_slices,
        shape=result.shape,
    )

    return UploadResponse(
        success=True,
        job_id=job_id,
        num_slices=result.num_slices,
        shape=result.shape,
        has_real_pet=real_pet_path is not None,
    )


@app.get("/cases/{job_id}/meta", response_model=CaseMetaResponse)
async def get_case_meta(job_id: str) -> CaseMetaResponse:
    record = case_records.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return CaseMetaResponse(
        success=True,
        job_id=record.job_id,
        num_slices=record.num_slices,
        shape=record.shape,
        has_real_pet=record.real_pet_path is not None,
    )


@app.get("/cases/{job_id}/slice/{index}")
async def get_case_slice(
    job_id: str,
    index: int,
    view: Literal["ct", "real_pet", "pred_pet"] = Query(...),
) -> Response:
    if converter is None:
        raise HTTPException(status_code=500, detail="Converter is not initialized")

    record = case_records.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if index < 0 or index >= record.num_slices:
        raise HTTPException(status_code=400, detail="Slice index out of range")

    if view == "ct":
        volume_path = record.ct_path
    elif view == "real_pet":
        if not record.real_pet_path:
            raise HTTPException(
                status_code=404, detail="Real PET not provided for this case"
            )
        volume_path = record.real_pet_path
    else:
        volume_path = record.pred_pet_path

    image_bytes = converter.get_slice_png(volume_path, index, view)
    return Response(content=image_bytes, media_type="image/png")
