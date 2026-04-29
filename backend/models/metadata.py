from typing import Literal, Optional

from pydantic import BaseModel, Field


class CaseMetaResponse(BaseModel):
    success: bool
    job_id: str
    source_type: Literal["nifti", "dicom_dir"]
    upload_mode: Literal["inference_only", "with_evaluation"]
    modality: Optional[str] = Field(
        default=None, description="Imaging modality, typically CT"
    )
    num_slices: int
    shape: list[int]
    has_real_pet: bool
    spacing_xyz_mm: Optional[list[float]] = Field(
        default=None, description="Voxel spacing in X/Y/Z (mm)"
    )
    slice_spacing_mm: Optional[float] = Field(
        default=None, description="Slice spacing in mm when available"
    )
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_id: Optional[str] = None
    study_date: Optional[str] = None
    processing_status: Literal["pending", "processing", "completed", "failed"]
    processing_error: Optional[str] = None
