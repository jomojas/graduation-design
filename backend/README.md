## Backend Service

This FastAPI backend runs 2.5D CT-to-PET inference using a pretrained generator.

### Input/Output

- Input CT volume: `.nii` or `.nii.gz`
- Optional real PET volume: `.nii` or `.nii.gz` (evaluation mode)
- Inference: 7-slice sliding window with edge-replication padding (`+3` on each side)
- Output predicted PET volume is saved as NIfTI and served slice-by-slice as PNG

### Run

```bash
uv sync
uv run python run_server.py
```

### Endpoints

- `GET /status`
- `POST /upload` with multipart fields:
  - `ct_file` (required)
  - `real_pet_file` (optional)
- `GET /cases/{job_id}/meta`
- `GET /cases/{job_id}/slice/{index}?view=ct|real_pet|pred_pet`

### Checkpoint

Place pretrained weights at:

- `backend/checkpoints/generator.pth`
