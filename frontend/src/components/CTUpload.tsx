import React, { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import NiivueVolumeViewer, { type NiivueVolumeViewerProps } from './NiivueVolumeViewer'
import type { NiivueViewerLike } from '../hooks/useNiivueViewer'

const API_BASE = 'http://localhost:8000'

type CaseMeta = {
  job_id: string
  study_id?: string
  source_type: 'nifti' | 'dicom_zip' | 'dicom_dir'
  num_slices: number
  shape: [number, number, number]
  has_real_pet: boolean
}

type StudyResultMetrics = {
  inference_time_ms: number | null
  output_shape: number[] | null
  slices_processed: number | null
  psnr: number | null
  ssim: number | null
  evaluation_status: 'completed' | 'skipped' | 'failed' | null
  evaluation_reason: string | null
}

type StudyResultVolume = {
  available: boolean
  nifti_path: string | null
}

type StudyResultResponse = {
  success: boolean
  study_id: string
  job_id: string
  has_real_pet: boolean
  num_slices: number
  shape: number[]
  metrics: StudyResultMetrics
  ct: StudyResultVolume
  predicted_pet: StudyResultVolume
  real_pet: StudyResultVolume
}

type UploadMode = 'nifti' | 'dicom_zip' | 'dicom_dir'
type UploadStage = 'idle' | 'uploading' | 'processing' | 'rendering'

const MAX_MB = 200

const fileSizeOk = (file: File) => file.size / 1024 / 1024 <= MAX_MB

const formatMetric = (value: number | null, precision: number) => {
  if (value === null || value === undefined) {
    return 'N/A'
  }
  return value.toFixed(precision)
}

const CTUpload: React.FC = () => {
  const [uploadMode, setUploadMode] = useState<UploadMode>('nifti')
  const [niftiCtFile, setNiftiCtFile] = useState<File | null>(null)
  const [realPetFile, setRealPetFile] = useState<File | null>(null)
  const [dicomZipFile, setDicomZipFile] = useState<File | null>(null)
  const [dicomDirFiles, setDicomDirFiles] = useState<File[]>([])
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStage, setUploadStage] = useState<UploadStage>('idle')
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [caseMeta, setCaseMeta] = useState<CaseMeta | null>(null)
  const [studyResult, setStudyResult] = useState<StudyResultResponse | null>(null)
  const [sliceIndex, setSliceIndex] = useState(0)
  const [petColormap, setPetColormap] = useState<'hot' | 'plasma'>('hot')
  const [fusionOpacity, setFusionOpacity] = useState(65)
  const [ctViewer, setCtViewer] = useState<NiivueViewerLike | null>(null)
  const [predViewer, setPredViewer] = useState<NiivueViewerLike | null>(null)
  const [realViewer, setRealViewer] = useState<NiivueViewerLike | null>(null)

  useEffect(() => {
    checkBackendStatus()
  }, [])

  const checkBackendStatus = async () => {
    try {
      await axios.get(`${API_BASE}/status`)
      setBackendStatus(true)
    } catch {
      setBackendStatus(false)
    }
  }

  const extractBackendError = (error: unknown): string => {
    const maybeAxiosError = error as {
      response?: {
        status?: number
        data?: {
          detail?: unknown
        } | string
      }
    }
    const hasAxiosGuard =
      typeof (axios as unknown as { isAxiosError?: (value: unknown) => boolean }).isAxiosError ===
      'function'
    const isAxiosLikeError = hasAxiosGuard
      ? (axios as unknown as { isAxiosError: (value: unknown) => boolean }).isAxiosError(error)
      : typeof maybeAxiosError === 'object' && maybeAxiosError !== null && 'response' in maybeAxiosError

    if (isAxiosLikeError) {
      const detail =
        maybeAxiosError.response?.data && typeof maybeAxiosError.response.data === 'object'
          ? maybeAxiosError.response.data.detail
          : undefined
      if (Array.isArray(detail)) {
        const joined = detail
          .map((item) => {
            if (typeof item === 'string') {
              return item
            }
            if (item && typeof item === 'object' && 'msg' in item) {
              return String(item.msg)
            }
            return JSON.stringify(item)
          })
          .join('; ')
        if (joined) {
          return joined
        }
      }
      if (typeof detail === 'string' && detail.trim().length > 0) {
        return detail
      }
      if (typeof maybeAxiosError.response?.data === 'string') {
        return maybeAxiosError.response.data
      }
      if (maybeAxiosError.response?.status) {
        return `Request failed with status ${maybeAxiosError.response.status}`
      }
    }
    return 'Upload or inference failed'
  }

  const runDisabled = useMemo(() => {
    if (uploadMode === 'nifti') {
      return !niftiCtFile
    }
    if (uploadMode === 'dicom_zip') {
      return !dicomZipFile
    }
    return dicomDirFiles.length === 0
  }, [dicomDirFiles.length, dicomZipFile, niftiCtFile, uploadMode])

  const stageStep = useMemo(() => {
    if (uploadStage === 'uploading') {
      return 0
    }
    if (uploadStage === 'processing') {
      return 1
    }
    if (uploadStage === 'rendering') {
      return 2
    }
    return -1
  }, [uploadStage])

  const activeResult = studyResult

  const resolveVolumeUrl = useCallback((niftiPath: string | null | undefined) => {
    if (!niftiPath) {
      return null
    }
    if (niftiPath.startsWith('http://') || niftiPath.startsWith('https://')) {
      return niftiPath
    }
    return `${API_BASE}${niftiPath}`
  }, [])

  const ctVolumeUrl = useMemo(
    () => resolveVolumeUrl(activeResult?.ct?.nifti_path),
    [activeResult?.ct?.nifti_path, resolveVolumeUrl]
  )
  const predPetVolumeUrl = useMemo(
    () => resolveVolumeUrl(activeResult?.predicted_pet?.nifti_path),
    [activeResult?.predicted_pet?.nifti_path, resolveVolumeUrl]
  )
  const realPetVolumeUrl = useMemo(
    () => resolveVolumeUrl(activeResult?.real_pet?.nifti_path),
    [activeResult?.real_pet?.nifti_path, resolveVolumeUrl]
  )

  const ctVolumes = useMemo<NonNullable<NiivueVolumeViewerProps['volumes']>>(() => {
    if (!ctVolumeUrl) {
      return []
    }
    return [{ url: ctVolumeUrl }]
  }, [ctVolumeUrl])

  const realPetVolumes = useMemo<NonNullable<NiivueVolumeViewerProps['volumes']>>(() => {
    if (!realPetVolumeUrl) {
      return []
    }
    return [{ url: realPetVolumeUrl, colormap: petColormap }]
  }, [petColormap, realPetVolumeUrl])

  const predictedFusionVolumes = useMemo<NonNullable<NiivueVolumeViewerProps['volumes']>>(() => {
    if (!ctVolumeUrl || !predPetVolumeUrl) {
      return []
    }
    return [
      { url: ctVolumeUrl },
      { url: predPetVolumeUrl, colormap: petColormap, opacity: fusionOpacity / 100 }
    ]
  }, [ctVolumeUrl, fusionOpacity, petColormap, predPetVolumeUrl])

  const hasRealPet = Boolean(activeResult?.has_real_pet && realPetVolumeUrl)
  const maxSliceIndex = Math.max((activeResult?.num_slices ?? caseMeta?.num_slices ?? 1) - 1, 0)

  const hasMetrics = useMemo(() => {
    const metrics = activeResult?.metrics
    if (!metrics) {
      return false
    }
    return [
      metrics.inference_time_ms,
      metrics.slices_processed,
      metrics.psnr,
      metrics.ssim,
      metrics.evaluation_status,
      metrics.evaluation_reason
    ].some((value) => value !== null && value !== undefined)
  }, [activeResult?.metrics])

  const metadataShape = activeResult?.shape ?? caseMeta?.shape
  const metadataSlices = activeResult?.num_slices ?? caseMeta?.num_slices
  const metadataStudyId = activeResult?.study_id ?? caseMeta?.study_id ?? null
  const metadataJobId = activeResult?.job_id ?? caseMeta?.job_id ?? null
  const metadataSource = caseMeta?.source_type ?? 'nifti'
  const metadataHasRealPet = activeResult?.has_real_pet ?? caseMeta?.has_real_pet ?? false

  const metricsTone = activeResult?.metrics?.evaluation_status === 'failed' ? 'warning' : 'success'
  const processingHeadline = uploading
    ? 'Study pipeline is running'
    : uploadError
      ? 'Last run failed'
      : activeResult
        ? 'Last run completed'
        : 'No processing started yet'
  const processingMessage = uploading
    ? 'Upload, inference, and workspace rendering are in progress.'
    : uploadError
      ? 'Inspect the upload failure panel for backend details and retry the study.'
      : activeResult
        ? 'Outputs are ready for synchronized review.'
        : 'Submit a study to populate state cards and metrics.'
  const processingTone = uploadError ? 'error' : activeResult ? 'success' : uploading ? 'info' : 'warning'

  const ctSyncPeers = useMemo(() => {
    const peers = [predViewer, realViewer]
    return peers.filter((peer): peer is NiivueViewerLike => Boolean(peer))
  }, [predViewer, realViewer])

  const predSyncPeers = useMemo(() => {
    const peers = [ctViewer, realViewer]
    return peers.filter((peer): peer is NiivueViewerLike => Boolean(peer))
  }, [ctViewer, realViewer])

  const realSyncPeers = useMemo(() => {
    const peers = [ctViewer, predViewer]
    return peers.filter((peer): peer is NiivueViewerLike => Boolean(peer))
  }, [ctViewer, predViewer])

  const handleCtViewerReady = useCallback((viewer: NiivueViewerLike) => {
    setCtViewer((previous) => (previous === viewer ? previous : viewer))
  }, [])

  const handlePredViewerReady = useCallback((viewer: NiivueViewerLike) => {
    setPredViewer((previous) => (previous === viewer ? previous : viewer))
  }, [])

  const handleRealViewerReady = useCallback((viewer: NiivueViewerLike) => {
    setRealViewer((previous) => (previous === viewer ? previous : viewer))
  }, [])

  useEffect(() => {
    if (!hasRealPet && realViewer) {
      setRealViewer(null)
    }
  }, [hasRealPet, realViewer])

  useEffect(() => {
    if (sliceIndex > maxSliceIndex) {
      setSliceIndex(maxSliceIndex)
    }
  }, [maxSliceIndex, sliceIndex])

  const validateNiftiFile = (file: File) => {
    const lower = file.name.toLowerCase()
    const isNifti = lower.endsWith('.nii') || lower.endsWith('.nii.gz')
    if (!isNifti) {
      return 'Only .nii or .nii.gz files are supported'
    }
    if (!fileSizeOk(file)) {
      return 'File size must be <= 200MB'
    }
    return null
  }

  const validateZipFile = (file: File) => {
    const lower = file.name.toLowerCase()
    if (!lower.endsWith('.zip')) {
      return 'Only .zip archives are supported in zipped DICOM mode'
    }
    if (!fileSizeOk(file)) {
      return 'ZIP file size must be <= 200MB'
    }
    return null
  }

  const validateDicomDirectoryFiles = (files: File[]) => {
    const tooLarge = files.find((file) => !fileSizeOk(file))
    if (tooLarge) {
      return 'Each directory file must be <= 200MB'
    }
    return null
  }

  const handleUploadAndInfer = async () => {
    if (uploadMode === 'nifti' && !niftiCtFile) {
      setUploadError('Please upload a CT NIfTI file')
      return
    }
    if (uploadMode === 'dicom_zip' && !dicomZipFile) {
      setUploadError('Please upload a DICOM ZIP file')
      return
    }
    if (uploadMode === 'dicom_dir' && dicomDirFiles.length === 0) {
      setUploadError('Please choose a DICOM directory')
      return
    }

    setUploading(true)
    setUploadStage('uploading')
    setUploadError(null)
    setCaseMeta(null)
    setStudyResult(null)
    setSliceIndex(0)

    try {
      const formData = new FormData()
      if (uploadMode === 'nifti' && niftiCtFile) {
        formData.append('ct_file', niftiCtFile)
        if (realPetFile) {
          formData.append('real_pet_file', realPetFile)
        }
      }
      if (uploadMode === 'dicom_zip' && dicomZipFile) {
        formData.append('ct_file', dicomZipFile)
      }
      if (uploadMode === 'dicom_dir') {
        dicomDirFiles.forEach((fileObj) => {
          const relativePath = (fileObj as File & { webkitRelativePath?: string }).webkitRelativePath
          formData.append('dicom_files', fileObj, relativePath || fileObj.name)
        })
      }

      setUploadStage('processing')
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const uploadMeta = response.data as CaseMeta
      setUploadStage('rendering')
      setCaseMeta(uploadMeta)

      const studyId = uploadMeta.study_id || uploadMeta.job_id
      const studyResultResponse = await axios.get(`${API_BASE}/studies/${studyId}/result`)
      setStudyResult(studyResultResponse.data as StudyResultResponse)
    } catch (error: unknown) {
      const errorMessage = extractBackendError(error)
      setUploadError(errorMessage)
    } finally {
      setUploading(false)
      setUploadStage('idle')
    }
  }

  const handleReset = () => {
    setNiftiCtFile(null)
    setRealPetFile(null)
    setDicomZipFile(null)
    setDicomDirFiles([])
    setUploadError(null)
    setUploadStage('idle')
    setCaseMeta(null)
    setStudyResult(null)
    setSliceIndex(0)
    setPetColormap('hot')
    setFusionOpacity(65)
  }

  return (
    <div className="medical-dashboard-grid">
      <aside className="medical-sidebar-stack">
        <section className="medical-panel">
          <h2 className="medical-panel-title">Backend Status</h2>
          <div className="medical-inline-cluster">
            <span className={`medical-status-pill medical-status-${backendStatus ? 'online' : 'offline'}`}>
              {backendStatus === null ? 'Checking...' : backendStatus ? 'Online' : 'Offline'}
            </span>
            <button type="button" className="medical-button medical-button-ghost" onClick={checkBackendStatus}>
              Refresh
            </button>
          </div>
        </section>

        <section className="medical-panel">
          <h2 className="medical-panel-title">Upload Volumes</h2>
          <p style={{ marginBottom: 8 }}>Submission mode</p>

          <fieldset className="medical-radio-group">
            <label className="medical-radio-option">
              <input
                type="radio"
                name="upload-mode"
                value="nifti"
                checked={uploadMode === 'nifti'}
                onChange={(event) => {
                  setUploadMode(event.target.value as UploadMode)
                  setUploadError(null)
                }}
              />
              <span>NIfTI</span>
            </label>
            <label className="medical-radio-option">
              <input
                type="radio"
                name="upload-mode"
                value="dicom_zip"
                checked={uploadMode === 'dicom_zip'}
                onChange={(event) => {
                  setUploadMode(event.target.value as UploadMode)
                  setUploadError(null)
                }}
              />
              <span>ZIP DICOM</span>
            </label>
            <label className="medical-radio-option">
              <input
                type="radio"
                name="upload-mode"
                value="dicom_dir"
                checked={uploadMode === 'dicom_dir'}
                onChange={(event) => {
                  setUploadMode(event.target.value as UploadMode)
                  setUploadError(null)
                }}
              />
              <span>Directory DICOM</span>
            </label>
          </fieldset>

          {uploadMode === 'nifti' && (
            <>
              <label className="medical-file-control">
                <span>CT NIfTI (required)</span>
                <input
                  data-testid="ct-nifti-input"
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null
                    if (!file) {
                      setNiftiCtFile(null)
                      return
                    }
                    const error = validateNiftiFile(file)
                    if (error) {
                      setUploadError(error)
                      setNiftiCtFile(null)
                      return
                    }
                    setUploadError(null)
                    setNiftiCtFile(file)
                  }}
                />
                <small>{niftiCtFile ? niftiCtFile.name : 'No file selected'}</small>
              </label>

              <label className="medical-file-control">
                <span>Real PET NIfTI (optional)</span>
                <input
                  data-testid="real-pet-input"
                  type="file"
                  accept=".nii,.nii.gz"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null
                    if (!file) {
                      setRealPetFile(null)
                      return
                    }
                    const error = validateNiftiFile(file)
                    if (error) {
                      setUploadError(error)
                      setRealPetFile(null)
                      return
                    }
                    setUploadError(null)
                    setRealPetFile(file)
                  }}
                />
                <small>{realPetFile ? realPetFile.name : 'No file selected'}</small>
              </label>
            </>
          )}

          {uploadMode === 'dicom_zip' && (
            <label className="medical-file-control">
              <span>DICOM ZIP (required)</span>
              <input
                data-testid="dicom-zip-input"
                type="file"
                accept=".zip"
                onChange={(event) => {
                  const file = event.target.files?.[0] ?? null
                  if (!file) {
                    setDicomZipFile(null)
                    return
                  }
                  const error = validateZipFile(file)
                  if (error) {
                    setUploadError(error)
                    setDicomZipFile(null)
                    return
                  }
                  setUploadError(null)
                  setDicomZipFile(file)
                }}
              />
              <small>{dicomZipFile ? dicomZipFile.name : 'No file selected'}</small>
            </label>
          )}

          {uploadMode === 'dicom_dir' && (
            <label className="medical-file-control">
              <span>DICOM directory (required)</span>
              <input
                data-testid="dicom-dir-input"
                type="file"
                multiple
                {...({ webkitdirectory: 'true', directory: '' } as React.InputHTMLAttributes<HTMLInputElement>)}
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? [])
                  if (files.length === 0) {
                    setDicomDirFiles([])
                    return
                  }
                  const error = validateDicomDirectoryFiles(files)
                  if (error) {
                    setUploadError(error)
                    setDicomDirFiles([])
                    return
                  }
                  setUploadError(null)
                  setDicomDirFiles(files)
                }}
              />
              <small>
                {dicomDirFiles.length > 0
                  ? `${dicomDirFiles.length} files ready for upload`
                  : 'Choose a folder containing DICOM slices'}
              </small>
            </label>
          )}

          {uploadError && (
            <div className="medical-banner medical-banner-error" style={{ marginTop: 16 }}>
              <div className="medical-banner-title">Upload failed</div>
              <div>{uploadError}</div>
            </div>
          )}

          <div className="medical-actions-row">
            <button
              type="button"
              className="medical-button medical-button-primary"
              onClick={handleUploadAndInfer}
              disabled={runDisabled || uploading}
            >
              {uploading ? 'Running...' : 'Run 2.5D Inference'}
            </button>
            <button
              type="button"
              className="medical-button medical-button-secondary"
              onClick={handleReset}
              disabled={uploading}
            >
              Reset
            </button>
          </div>
        </section>

        <section className="medical-panel">
          <h2 className="medical-panel-title">Processing State</h2>
          <div className={`medical-banner medical-banner-${processingTone}`}>
            <div className="medical-banner-title">{processingHeadline}</div>
            <div>{processingMessage}</div>
          </div>
          <ol className="medical-steps-list">
            {['Upload payload', 'Run inference', 'Prepare results'].map((title, index) => {
              const status = stageStep >= index ? 'done' : stageStep === index - 1 ? 'active' : 'todo'
              return (
                <li key={title} className={`medical-step-item medical-step-${status}`}>
                  <span className="medical-step-index">{index + 1}</span>
                  <span>{title}</span>
                </li>
              )
            })}
          </ol>
        </section>

        <section className="medical-panel">
          <h2 className="medical-panel-title">Study Metadata</h2>
          <div className="medical-meta-grid">
            <div className="medical-meta-item">
              <div className="medical-meta-label">Study ID</div>
              <div className="medical-meta-value">{metadataStudyId ?? 'Pending submission'}</div>
            </div>
            <div className="medical-meta-item">
              <div className="medical-meta-label">Job ID</div>
              <div className="medical-meta-value">{metadataJobId ?? 'Pending submission'}</div>
            </div>
            <div className="medical-meta-item">
              <div className="medical-meta-label">Source</div>
              <div className="medical-meta-value">{metadataSource}</div>
            </div>
            <div className="medical-meta-item">
              <div className="medical-meta-label">Slices</div>
              <div className="medical-meta-value">{metadataSlices ?? 'N/A'}</div>
            </div>
            <div className="medical-meta-item">
              <div className="medical-meta-label">Volume Shape</div>
              <div className="medical-meta-value">
                {metadataShape && metadataShape.length > 0 ? metadataShape.join(' x ') : 'N/A'}
              </div>
            </div>
            <div className="medical-meta-item">
              <div className="medical-meta-label">Real PET</div>
              <div className="medical-meta-value">{metadataHasRealPet ? 'Provided' : 'Not provided'}</div>
            </div>
          </div>
        </section>

        {hasMetrics && activeResult && (
          <section className="medical-panel">
            <h2 className="medical-panel-title">Metrics</h2>
            <div className="medical-metrics-grid">
              <div className="medical-metric-item">
                <div className="medical-metric-label">Inference (ms)</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.inference_time_ms, 1)}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">Slices</div>
                <div className="medical-metric-value">{activeResult.metrics.slices_processed ?? 'N/A'}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">PSNR</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.psnr, 3)}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">SSIM</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.ssim, 4)}</div>
              </div>
            </div>
            {activeResult.metrics.evaluation_status && (
              <div
                className={`medical-banner ${
                  activeResult.metrics.evaluation_status === 'failed'
                    ? 'medical-banner-error'
                    : 'medical-banner-info'
                }`}
                style={{ marginTop: 12 }}
              >
                <div>
                  Evaluation: {activeResult.metrics.evaluation_status}
                  {activeResult.metrics.evaluation_reason
                    ? ` (${activeResult.metrics.evaluation_reason})`
                    : ''}
                </div>
              </div>
            )}
          </section>
        )}

        <section className="medical-panel">
          <h2 className="medical-panel-title">Mode Notes</h2>
          <div className="medical-banner medical-banner-info">
            <div className="medical-banner-title">Evaluation Mode</div>
            <div>NIfTI mode supports optional Real PET. ZIP and directory DICOM modes run inference-only.</div>
          </div>
          <div style={{ marginTop: 12, color: '#5b6473' }}>
            Backend `/upload` accepts ct_file (.nii/.nii.gz/.zip) or repeated dicom_files.
          </div>
          <div style={{ marginTop: 6, color: '#5b6473' }}>
            PNG slice endpoints remain backend compatibility-only; the workspace renders from study result NIfTI paths.
          </div>
        </section>
      </aside>

      <section className="medical-main-stack">
        <section className="medical-panel medical-workspace-panel">
          <div className="medical-workspace-header">
            <h2 className="medical-panel-title">Synchronized Imaging Workspace</h2>
            {activeResult && (
              <div className="medical-workspace-header-meta">
                Slices: {activeResult.num_slices} | Shape: {activeResult.shape.join(' x ')}
              </div>
            )}
          </div>

          {uploadError && (
            <div className="medical-banner medical-banner-error" style={{ marginBottom: 16 }}>
              <div className="medical-banner-title">Processing failed</div>
              <div>Pipeline halted before viewer synchronization. Resolve the failure reason in the sidebar and retry.</div>
            </div>
          )}

          {activeResult && !uploadError && (
            <div
              className={`medical-banner ${metricsTone === 'warning' ? 'medical-banner-warning' : 'medical-banner-success'}`}
              style={{ marginBottom: 16 }}
            >
              <div className="medical-banner-title">Study workspace is ready</div>
              <div>CT, predicted PET fusion, and optional reference PET are synchronized by slice index.</div>
            </div>
          )}

          {!activeResult && (
            <div className="medical-workspace-empty-state">
              {uploading ? (
                <div className="medical-loading-block" aria-label="Loading workspace">
                  <div className="medical-loading-line" />
                  <div className="medical-loading-line" />
                  <div className="medical-loading-line" />
                </div>
              ) : (
                <div className="medical-empty-state">Submit a study to start</div>
              )}
            </div>
          )}

          {activeResult && (
            <div>
              <div className="medical-subpanel" style={{ marginBottom: 16 }}>
                <h3 className="medical-subpanel-title">Fusion Controls</h3>
                <div className="medical-fusion-controls">
                  <div>
                    <div style={{ marginBottom: 8, color: '#596273' }}>PET colormap</div>
                    <div className="medical-toggle-group" role="group" aria-label="PET colormap">
                      <button
                        type="button"
                        className={`medical-toggle ${petColormap === 'hot' ? 'medical-toggle-active' : ''}`}
                        onClick={() => setPetColormap('hot')}
                      >
                        hot
                      </button>
                      <button
                        type="button"
                        className={`medical-toggle ${petColormap === 'plasma' ? 'medical-toggle-active' : ''}`}
                        onClick={() => setPetColormap('plasma')}
                      >
                        plasma
                      </button>
                    </div>
                  </div>
                  <div>
                    <label htmlFor="fusion-opacity" style={{ marginBottom: 8, color: '#596273', display: 'block' }}>
                      Fusion opacity: {fusionOpacity}%
                    </label>
                    <input
                      id="fusion-opacity"
                      type="range"
                      min={0}
                      max={100}
                      value={fusionOpacity}
                      onChange={(event) => setFusionOpacity(Number(event.target.value))}
                    />
                  </div>
                </div>
              </div>

              <div className={`medical-viewer-grid ${hasRealPet ? 'medical-viewer-grid-real' : ''}`}>
                <article className="medical-subpanel">
                  <h3 className="medical-subpanel-title">CT Volume</h3>
                  <div className="medical-viewer-frame">
                    <NiivueVolumeViewer
                      volumes={ctVolumes}
                      sliceIndex={sliceIndex}
                      sliceCount={activeResult.num_slices}
                      syncPeers={ctSyncPeers}
                      onViewerReady={handleCtViewerReady}
                    />
                  </div>
                </article>

                {hasRealPet && (
                  <article className="medical-subpanel">
                    <h3 className="medical-subpanel-title">Real PET Reference</h3>
                    <div className="medical-viewer-frame">
                      <NiivueVolumeViewer
                        volumes={realPetVolumes}
                        sliceIndex={sliceIndex}
                        sliceCount={activeResult.num_slices}
                        syncPeers={realSyncPeers}
                        onViewerReady={handleRealViewerReady}
                      />
                    </div>
                  </article>
                )}

                <article className="medical-subpanel">
                  <h3 className="medical-subpanel-title">Predicted PET Fusion</h3>
                  <div className="medical-viewer-frame">
                    <NiivueVolumeViewer
                      volumes={predictedFusionVolumes}
                      sliceIndex={sliceIndex}
                      sliceCount={activeResult.num_slices}
                      syncPeers={predSyncPeers}
                      onViewerReady={handlePredViewerReady}
                    />
                  </div>
                </article>
              </div>

              <div style={{ marginTop: 20 }}>
                <label htmlFor="slice-index" style={{ marginBottom: 8, color: '#596273', display: 'block' }}>
                  Slice index: {sliceIndex}
                </label>
                <input
                  id="slice-index"
                  type="range"
                  min={0}
                  max={maxSliceIndex}
                  value={sliceIndex}
                  onChange={(event) => setSliceIndex(Number(event.target.value))}
                />
              </div>
            </div>
          )}
        </section>
      </section>
    </div>
  )
}

export default CTUpload
