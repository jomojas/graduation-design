# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## COMMANDS
USE THESE TO INTERACT WITH THE REPOSITORY

### FRONTEND (React)
```bash
# START DEV SERVER
npm run dev

# BUILD PRODUCTION BUILD
npm run build

# RUN PRODUCTION SERVER
npm start
```

### BACKEND (FastAPI)
```bash
# START API SERVER
uvicorn backend.main:app --reload

# RUN API TESTS
pytest
```

## CODE ARCHITECTURE

### SYSTEM STRUCTURE
```
C:\Users\Jomo\Desktop\folders\graduation-design
├── backend/
│   ├── main.py          # FastAPI API root
│   ├── models/          # Data models and validation
│   └── templates/       # HTML templates (if any)
├── frontend/
│   ├── src/
│   │   ├── App.tsx      # Root component
│   │   ├── components/
│   │   │   ├── CTUpload.tsx       # Primary UI component
│   │   │   └── DashboardShell.tsx
│   │   └── styles/
│   │       ├── globals.css        # Global styles
│   │       └── medical-buttons.css # Custom designs
│   └── package.json     # Frontend dependencies
└── docker-compose.yml   # Deployment orchestration
```

### KEY COMPONENTS
1. **CTUpload.tsx** - Handles file upload mode selection (NIfTI/ZIP/DICOM) and manages upload workflow
2. **DashboardShell.tsx** - Main layout container with status panels and workspace preview
3. **NiivueVolumeViewer** integration - Visualizes CT inputs and model outputs together
4. **API Services** - `/upload` and `/studies/{id}/result` endpoints

### DATA FLOW
FRONTEND → BACKEND
```
UPLOAD FILES → POST /upload → PROCESSING → GET /studies/{id}/result → RETURN METRICS
```

## KEY CONCEPTS
1. **Upload Workflow**
   - Multiple input modes (NIfTI/ZIP/DICOM)
   - Validation pipeline before API submission
   - Two-phase dissemination (upload → workflow processing → result retrieval)

2. **Visual Design System**
   - Neon aquamarine (#00F5F0) primary color
   - Glowing coral (#FF6B6B) accent color
   - Space black (#0F172A) background palette
   - Animated transition system with pulse effects

3. **Response Handling**
   - `StudyResultResponse` contains metrics and volume references
   - Metadata includes evaluation status and technical specifications
   - Volume paths are resolved via `API_BASE` endpoint

## NOTABLE DEPENDENCIES
- **Backend**: `fastapi[all]`, `uvicorn`, `torch`, `NiivueVolumeViewer` integration
- **Frontend**: `react`, `axios`, `niivue`, `tailwindcss`, custom medical UI components
