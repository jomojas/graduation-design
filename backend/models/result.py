from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class StudyResultMetrics(BaseModel):
    inference_time_ms: Optional[float] = None
    output_shape: Optional[list[int]] = None
    slices_processed: Optional[int] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    evaluation_status: Optional[Literal["completed", "skipped", "failed"]] = None
    evaluation_reason: Optional[str] = None


class StudyResultVolume(BaseModel):
    available: bool
    nifti_path: Optional[str] = None
    slice_endpoint_template: Optional[str] = None


class StudyStatusResponse(BaseModel):
    success: bool
    study_id: str
    job_id: str
    source_type: Literal["nifti", "dicom_dir"]
    upload_mode: Literal["inference_only", "with_evaluation"]
    status: Literal["pending", "processing", "completed", "failed"]
    error: Optional[str] = None
    has_real_pet: bool
    num_slices: int
    shape: list[int]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: StudyResultMetrics


class StudyResultResponse(BaseModel):
    success: bool
    study_id: str
    job_id: str
    source_type: Literal["nifti", "dicom_dir"]
    upload_mode: Literal["inference_only", "with_evaluation"]
    status: Literal["pending", "processing", "completed", "failed"]
    error: Optional[str] = None
    has_real_pet: bool
    num_slices: int
    shape: list[int]
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_date: Optional[str] = None
    metrics: StudyResultMetrics
    ct: StudyResultVolume
    predicted_pet: StudyResultVolume
    real_pet: StudyResultVolume
