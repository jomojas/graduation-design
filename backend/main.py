import os
import uuid
import zipfile
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from nibabel.filebasedimages import ImageFileError
from pydantic import BaseModel, ConfigDict, Field

from models.metadata import CaseMetaResponse
from models.result import (
    StudyResultMetrics,
    StudyResultResponse,
    StudyResultVolume,
    StudyStatusResponse,
)
from services.metrics import compute_volume_metrics
from services.converter import CT2PETConverter, get_converter
from utils.file_utils import CHECKPOINT_DIR, OUTPUT_DIR, UPLOAD_DIR, ensure_directories
from utils.standardization import (
    align_reference_pet_to_ct,
    extract_nifti_geometry,
    standardize_nifti_to_niigz,
    standardize_dicom_ct,
    standardize_dicom_pet,
)


logger = logging.getLogger(__name__)


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
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    evaluation_status: Optional[Literal["completed", "skipped", "failed"]] = None
    evaluation_reason: Optional[str] = None


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

    geometry: Optional[dict[str, Any]] = Field(
        default=None,
        description="Standardized geometry metadata for CT and optional reference PET",
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
study_manifests: dict[str, StudyManifest] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loaded, converter
    ensure_directories()

    model_path = os.path.join(CHECKPOINT_DIR, "generator.pth")
    if os.path.exists(model_path):
        try:
            converter = get_converter(model_path)
            model_loaded = True
            logger.info("Model loaded: %s", model_path)
        except Exception as exc:
            logger.exception("Model load failed: %s", exc)
            model_loaded = False
    else:
        logger.warning("Warning: checkpoint not found at %s", model_path)
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

app.mount(
    "/uploads", StaticFiles(directory=UPLOAD_DIR, check_dir=False), name="uploads"
)
app.mount(
    "/outputs", StaticFiles(directory=OUTPUT_DIR, check_dir=False), name="outputs"
)


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class UploadResponse(BaseModel):
    success: bool
    job_id: str
    source_type: Literal["nifti", "dicom_zip", "dicom_dir"]
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


def _is_nifti_filename(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _extract_zip_to_directory(zip_path: Path, target_dir: Path) -> None:
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.namelist():
                destination = (target_dir / member).resolve()
                if not str(destination).startswith(str(target_dir.resolve())):
                    raise HTTPException(status_code=400, detail="Invalid ZIP contents")
            archive.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400, detail="Invalid DICOM ZIP archive"
        ) from exc


def _save_directory_upload(dicom_files: list[UploadFile], target_dir: Path) -> None:
    if not dicom_files:
        raise HTTPException(status_code=400, detail="No DICOM directory files provided")

    for upload in dicom_files:
        if not upload.filename:
            continue
        relative = Path(upload.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise HTTPException(status_code=400, detail="Invalid DICOM directory path")
        destination = target_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = upload.file.read()
        destination.write_bytes(content)


def _create_manifest(
    *,
    job_id: str,
    source_type: Literal["nifti", "dicom_zip", "dicom_dir"],
    ct_path: Path,
    pred_pet_path: Path,
    real_pet_path: Optional[Path],
    num_slices: int,
    shape: tuple[int, int, int],
    metadata: Optional[dict[str, Any]] = None,
    geometry: Optional[dict[str, Any]] = None,
) -> StudyManifest:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "source_type": source_type,
        "upload_mode": "with_evaluation" if real_pet_path else "inference_only",
        "ct_volume_path": str(ct_path),
        "real_pet_volume_path": str(real_pet_path) if real_pet_path else None,
        "pred_pet_volume_path": str(pred_pet_path),
        "num_slices": num_slices,
        "shape": shape,
    }
    if metadata:
        payload["patient_id"] = metadata.get("patient_id")
        payload["patient_name"] = metadata.get("patient_name")
        payload["study_id"] = metadata.get("study_id")
        payload["study_date"] = metadata.get("study_date")
    if geometry:
        payload["geometry"] = geometry
    return StudyManifest(**payload)


def _resolve_study_manifest(study_id: str) -> Optional[StudyManifest]:
    direct_match = study_manifests.get(study_id)
    if direct_match is not None:
        return direct_match

    matched = [
        manifest
        for manifest in study_manifests.values()
        if manifest.study_id == study_id
    ]
    if not matched:
        return None
    matched.sort(key=lambda item: item.updated_at, reverse=True)
    return matched[0]


def _canonical_study_id(manifest: StudyManifest) -> str:
    return manifest.study_id or manifest.job_id


def _build_study_metrics(manifest: StudyManifest) -> StudyResultMetrics:
    output_shape: Optional[list[int]] = None
    if manifest.metrics.output_shape is not None:
        output_shape = list(manifest.metrics.output_shape)
    return StudyResultMetrics(
        inference_time_ms=manifest.metrics.inference_time_ms,
        output_shape=output_shape,
        slices_processed=manifest.metrics.slices_processed,
        psnr=manifest.metrics.psnr,
        ssim=manifest.metrics.ssim,
        evaluation_status=manifest.metrics.evaluation_status,
        evaluation_reason=manifest.metrics.evaluation_reason,
    )


def _build_study_volume(
    *,
    path: Optional[str],
    job_id: str,
    view: Literal["ct", "pred_pet", "real_pet"],
) -> StudyResultVolume:
    available = path is not None
    slice_endpoint_template: Optional[str] = None
    public_nifti_path: Optional[str] = None
    if available:
        slice_endpoint_template = f"/cases/{job_id}/slice/{{index}}?view={view}"
        public_nifti_path = _to_public_nifti_path(path)
    return StudyResultVolume(
        available=available,
        nifti_path=public_nifti_path,
        slice_endpoint_template=slice_endpoint_template,
    )


def _to_public_nifti_path(path: str) -> str:
    normalized_path = Path(path)
    upload_root = Path(UPLOAD_DIR).resolve()
    output_root = Path(OUTPUT_DIR).resolve()
    resolved = normalized_path.resolve()

    try:
        relative_upload = resolved.relative_to(upload_root)
        return f"/uploads/{relative_upload.as_posix()}"
    except ValueError:
        pass

    try:
        relative_output = resolved.relative_to(output_root)
        return f"/outputs/{relative_output.as_posix()}"
    except ValueError:
        return resolved.as_posix()


def _ensure_converter_ready() -> CT2PETConverter:
    global converter, model_loaded

    if converter is not None:
        model_loaded = True
        return converter

    model_path = os.path.join(CHECKPOINT_DIR, "generator.pth")
    if not os.path.exists(model_path):
        model_loaded = False
        raise HTTPException(
            status_code=503,
            detail=f"Model checkpoint not found at {model_path}",
        )

    try:
        converter = get_converter(model_path)
    except Exception as exc:
        model_loaded = False
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {str(exc)}",
        ) from exc

    model_loaded = True
    return converter


@app.get("/", response_model=StatusResponse)
async def root() -> StatusResponse:
    device = converter.device if converter else "cpu"
    return StatusResponse(status="ok", model_loaded=model_loaded, device=device)


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    device = converter.device if converter else "cpu"
    return StatusResponse(status="running", model_loaded=model_loaded, device=device)


@app.get("/studies/{study_id}/status", response_model=StudyStatusResponse)
async def get_study_status(study_id: str) -> StudyStatusResponse:
    manifest = _resolve_study_manifest(study_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Study not found")

    return StudyStatusResponse(
        success=True,
        study_id=_canonical_study_id(manifest),
        job_id=manifest.job_id,
        source_type=manifest.source_type,
        upload_mode=manifest.upload_mode,
        status=manifest.inference_status.state,
        error=manifest.inference_status.error or manifest.error_status,
        has_real_pet=manifest.real_pet_volume_path is not None,
        num_slices=manifest.num_slices,
        shape=list(manifest.shape),
        created_at=manifest.created_at,
        updated_at=manifest.updated_at,
        started_at=manifest.inference_status.started_at,
        completed_at=manifest.inference_status.completed_at,
        metrics=_build_study_metrics(manifest),
    )


@app.get("/studies/{study_id}/result", response_model=StudyResultResponse)
async def get_study_result(study_id: str) -> StudyResultResponse:
    manifest = _resolve_study_manifest(study_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Study not found")

    return StudyResultResponse(
        success=True,
        study_id=_canonical_study_id(manifest),
        job_id=manifest.job_id,
        source_type=manifest.source_type,
        upload_mode=manifest.upload_mode,
        status=manifest.inference_status.state,
        error=manifest.inference_status.error or manifest.error_status,
        has_real_pet=manifest.real_pet_volume_path is not None,
        num_slices=manifest.num_slices,
        shape=list(manifest.shape),
        patient_id=manifest.patient_id,
        patient_name=manifest.patient_name,
        study_date=manifest.study_date,
        metrics=_build_study_metrics(manifest),
        ct=_build_study_volume(
            path=manifest.ct_volume_path,
            job_id=manifest.job_id,
            view="ct",
        ),
        predicted_pet=_build_study_volume(
            path=manifest.pred_pet_volume_path,
            job_id=manifest.job_id,
            view="pred_pet",
        ),
        real_pet=_build_study_volume(
            path=manifest.real_pet_volume_path,
            job_id=manifest.job_id,
            view="real_pet",
        ),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_case(
    ct_file: Optional[UploadFile] = File(None),
    real_pet_file: Optional[UploadFile] = File(None),
    dicom_files: list[UploadFile] = File(default_factory=list),
    real_pet_dicom_files: list[UploadFile] = File(default_factory=list),
) -> UploadResponse:
    active_converter = _ensure_converter_ready()

    has_ct_file = ct_file is not None and bool(ct_file.filename)
    has_dicom_directory = len(dicom_files) > 0
    if not has_ct_file and not has_dicom_directory:
        raise HTTPException(
            status_code=400,
            detail="Provide either ct_file (.nii/.nii.gz/.zip/.dcm) or dicom_files",
        )

    if not has_dicom_directory and real_pet_dicom_files:
        raise HTTPException(
            status_code=400,
            detail="real_pet_dicom_files is only supported for Directory DICOM upload mode",
        )

    job_id = uuid.uuid4().hex
    upload_dir = Path(UPLOAD_DIR) / job_id
    output_dir = Path(OUTPUT_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_path: Path
    real_pet_path: Optional[Path] = None
    source_type: Literal["nifti", "dicom_zip", "dicom_dir"]
    dicom_metadata: Optional[dict[str, Any]] = None
    geometry_metadata: Optional[dict[str, Any]] = None

    if has_dicom_directory:
        source_type = "dicom_dir"
        if real_pet_file and real_pet_file.filename:
            raise HTTPException(
                status_code=400,
                detail="Use real_pet_dicom_files for optional PET in Directory DICOM mode",
            )
        dicom_dir = upload_dir / "dicom_dir"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        _save_directory_upload(dicom_files, dicom_dir)
        ct_path = upload_dir / "ct.nii.gz"
        standardized = standardize_dicom_ct(dicom_dir, ct_path)
        ct_path = standardized.ct_path
        dicom_metadata = standardized.metadata
        geometry_metadata = {"ct": standardized.geometry}

        if real_pet_dicom_files:
            pet_dicom_dir = upload_dir / "real_pet_dicom_dir"
            pet_dicom_dir.mkdir(parents=True, exist_ok=True)
            _save_directory_upload(real_pet_dicom_files, pet_dicom_dir)

            raw_pet_path = upload_dir / "real_pet_dicom.nii.gz"
            standardized_pet = standardize_dicom_pet(pet_dicom_dir, raw_pet_path)

            aligned_pet_path = upload_dir / "real_pet.nii.gz"
            real_pet_path, alignment_meta = align_reference_pet_to_ct(
                ct_path,
                standardized_pet.ct_path,
                aligned_pet_path,
            )
            geometry_metadata["reference_pet"] = {
                "geometry": extract_nifti_geometry(real_pet_path),
                "alignment": alignment_meta,
            }
    else:
        assert ct_file is not None
        filename = ct_file.filename or ""
        lower_name = filename.lower()

        if _is_nifti_filename(filename):
            source_type = "nifti"
            _validate_nifti(filename)
            if real_pet_file and real_pet_file.filename:
                _validate_nifti(real_pet_file.filename)

            ct_content = await ct_file.read()
            if len(ct_content) > 200 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="CT file exceeds 200MB")
            ct_ext = _get_nifti_extension(ct_content)
            raw_ct_path = upload_dir / f"ct_input{ct_ext}"
            raw_ct_path.write_bytes(ct_content)
            ct_path = upload_dir / "ct.nii.gz"
            try:
                standardize_nifti_to_niigz(raw_ct_path, ct_path)
                geometry_metadata = {"ct": extract_nifti_geometry(ct_path)}
            except (ImageFileError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"Invalid NIfTI file: {str(exc)}"
                ) from exc

            if real_pet_file and real_pet_file.filename:
                pet_content = await real_pet_file.read()
                if len(pet_content) > 200 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, detail="Real PET file exceeds 200MB"
                    )
                raw_pet_ext = _get_nifti_extension(pet_content)
                raw_pet_path = upload_dir / f"real_pet_input{raw_pet_ext}"
                raw_pet_path.write_bytes(pet_content)
                standardized_pet_path = upload_dir / "real_pet.nii.gz"
                real_pet_path, alignment_meta = align_reference_pet_to_ct(
                    ct_path,
                    raw_pet_path,
                    standardized_pet_path,
                )
                geometry_metadata["reference_pet"] = {
                    "geometry": extract_nifti_geometry(real_pet_path),
                    "alignment": alignment_meta,
                }
        elif lower_name.endswith(".zip") or lower_name.endswith(".dcm"):
            source_type = "dicom_zip" if lower_name.endswith(".zip") else "dicom_dir"
            dicom_root = upload_dir / "dicom"
            dicom_root.mkdir(parents=True, exist_ok=True)

            if lower_name.endswith(".zip"):
                archive_bytes = await ct_file.read()
                if len(archive_bytes) > 200 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, detail="DICOM ZIP file exceeds 200MB"
                    )
                archive_path = dicom_root / "study.zip"
                archive_path.write_bytes(archive_bytes)
                _extract_zip_to_directory(archive_path, dicom_root)
                archive_path.unlink(missing_ok=True)
            else:
                single_dcm = await ct_file.read()
                if len(single_dcm) > 200 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, detail="DICOM file exceeds 200MB"
                    )
                (dicom_root / "slice.dcm").write_bytes(single_dcm)

            ct_path = upload_dir / "ct.nii.gz"
            standardized = standardize_dicom_ct(dicom_root, ct_path)
            ct_path = standardized.ct_path
            dicom_metadata = standardized.metadata
            geometry_metadata = {"ct": standardized.geometry}

            if real_pet_file and real_pet_file.filename:
                _validate_nifti(real_pet_file.filename)
                pet_content = await real_pet_file.read()
                if len(pet_content) > 200 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, detail="Real PET file exceeds 200MB"
                    )
                raw_pet_ext = _get_nifti_extension(pet_content)
                raw_pet_path = upload_dir / f"real_pet_input{raw_pet_ext}"
                raw_pet_path.write_bytes(pet_content)
                standardized_pet_path = upload_dir / "real_pet.nii.gz"
                real_pet_path, alignment_meta = align_reference_pet_to_ct(
                    ct_path,
                    raw_pet_path,
                    standardized_pet_path,
                )
                geometry_metadata["reference_pet"] = {
                    "geometry": extract_nifti_geometry(real_pet_path),
                    "alignment": alignment_meta,
                }
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported CT upload format. Use .nii, .nii.gz, .zip, or DICOM directory files",
            )

    pred_pet_path = output_dir / "pred_pet.nii.gz"
    try:
        result = active_converter.convert_nifti(str(ct_path), str(pred_pet_path))
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

    manifest = _create_manifest(
        job_id=job_id,
        source_type=source_type,
        ct_path=ct_path,
        pred_pet_path=pred_pet_path,
        real_pet_path=real_pet_path,
        num_slices=result.num_slices,
        shape=result.shape,
        metadata=dicom_metadata,
        geometry=geometry_metadata,
    )
    manifest.inference_status = InferenceStatus(
        state=result.inference_status,
        started_at=result.inference_started_at,
        completed_at=result.inference_completed_at,
        error=result.inference_error,
    )
    metrics_result = compute_volume_metrics(
        pred_path=str(pred_pet_path),
        reference_path=str(real_pet_path) if real_pet_path else None,
    )
    manifest.metrics = MetricsState(
        inference_time_ms=result.inference_duration_ms,
        output_shape=result.shape,
        slices_processed=result.num_slices,
        psnr=metrics_result.psnr,
        ssim=metrics_result.ssim,
        evaluation_status=metrics_result.status,
        evaluation_reason=metrics_result.reason,
    )
    study_manifests[job_id] = manifest

    return UploadResponse(
        success=True,
        job_id=job_id,
        source_type=source_type,
        num_slices=result.num_slices,
        shape=result.shape,
        has_real_pet=real_pet_path is not None,
    )


@app.get("/cases/{job_id}/meta", response_model=CaseMetaResponse)
async def get_case_meta(job_id: str) -> CaseMetaResponse:
    manifest = study_manifests.get(job_id)
    if manifest is not None:
        ct_geometry: Optional[dict[str, Any]] = None
        if manifest.geometry:
            ct_geometry = manifest.geometry.get("ct")
        spacing_xyz_mm: Optional[list[float]] = None
        slice_spacing_mm: Optional[float] = None
        if isinstance(ct_geometry, dict):
            spacing_value = ct_geometry.get("spacing_xyz_mm")
            if isinstance(spacing_value, list):
                spacing_xyz_mm = [float(v) for v in spacing_value]
            slice_value = ct_geometry.get("slice_spacing_mm")
            if slice_value is not None:
                slice_spacing_mm = float(slice_value)
        return CaseMetaResponse(
            success=True,
            job_id=manifest.job_id,
            source_type=manifest.source_type,
            upload_mode=manifest.upload_mode,
            modality="CT",
            num_slices=manifest.num_slices,
            shape=list(manifest.shape),
            has_real_pet=manifest.real_pet_volume_path is not None,
            spacing_xyz_mm=spacing_xyz_mm,
            slice_spacing_mm=slice_spacing_mm,
            patient_id=manifest.patient_id,
            patient_name=manifest.patient_name,
            study_id=manifest.study_id,
            study_date=manifest.study_date,
            processing_status=manifest.inference_status.state,
            processing_error=manifest.inference_status.error or manifest.error_status,
        )

    record = case_records.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return CaseMetaResponse(
        success=True,
        job_id=record.job_id,
        source_type="nifti",
        upload_mode="with_evaluation" if record.real_pet_path else "inference_only",
        modality="CT",
        num_slices=record.num_slices,
        shape=list(record.shape),
        has_real_pet=record.real_pet_path is not None,
        spacing_xyz_mm=None,
        slice_spacing_mm=None,
        patient_id=None,
        patient_name=None,
        study_id=None,
        study_date=None,
        processing_status="completed",
        processing_error=None,
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
