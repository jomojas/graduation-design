import React, { useCallback, useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { Steps, Upload } from 'antd'
import { CheckCircleOutlined, CloudUploadOutlined, ExperimentOutlined } from '@ant-design/icons'
import type { UploadFile, UploadProps } from 'antd'
import NiivueVolumeViewer, { type NiivueVolumeViewerProps } from './NiivueVolumeViewer'
import type { NiivueViewerLike } from '../hooks/useNiivueViewer'
import { useLanguage } from '../i18n/LanguageProvider'

const API_BASE = 'http://localhost:8000'

type CaseMeta = {
  job_id: string
  study_id?: string
  source_type: 'nifti' | 'dicom_dir'
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

type UploadMode = 'nifti' | 'dicom_dir'
type UploadStage = 'idle' | 'uploading' | 'processing' | 'rendering'

const MAX_MB = 200

const fileSizeOk = (file: File) => file.size / 1024 / 1024 <= MAX_MB

const formatMetric = (value: number | null, precision: number, fallback: string) => {
  if (value === null || value === undefined) {
    return fallback
  }
  return value.toFixed(precision)
}

const CTUpload: React.FC = () => {
  const { t } = useLanguage()
  const PET_COLORMAP = 'hot'
  const FUSION_OPACITY = 0.65
  const [uploadMode, setUploadMode] = useState<UploadMode>('nifti')
  const [niftiCtFile, setNiftiCtFile] = useState<File | null>(null)
  const [realPetFile, setRealPetFile] = useState<File | null>(null)
  const [dicomDirFiles, setDicomDirFiles] = useState<File[]>([])
  const [realPetDicomDirFiles, setRealPetDicomDirFiles] = useState<File[]>([])
  const [niftiCtUploadList, setNiftiCtUploadList] = useState<UploadFile[]>([])
  const [realPetUploadList, setRealPetUploadList] = useState<UploadFile[]>([])
  const [dicomDirUploadList, setDicomDirUploadList] = useState<UploadFile[]>([])
  const [realPetDicomDirUploadList, setRealPetDicomDirUploadList] = useState<UploadFile[]>([])
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStage, setUploadStage] = useState<UploadStage>('idle')
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [studyResult, setStudyResult] = useState<StudyResultResponse | null>(null)
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
        return t('requestFailedWithStatus', maybeAxiosError.response.status)
      }
    }
    return t('uploadOrInferenceFailed')
  }

  const runDisabled = useMemo(() => {
    if (uploadMode === 'nifti') {
      return !niftiCtFile
    }
    return dicomDirFiles.length === 0
  }, [dicomDirFiles.length, niftiCtFile, uploadMode])

  const hasPayloadSelected = !runDisabled

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
    return [{ url: realPetVolumeUrl, colormap: PET_COLORMAP }]
  }, [realPetVolumeUrl])

  const predictedFusionVolumes = useMemo<NonNullable<NiivueVolumeViewerProps['volumes']>>(() => {
    if (!ctVolumeUrl || !predPetVolumeUrl) {
      return []
    }
    return [
      { url: ctVolumeUrl },
      { url: predPetVolumeUrl, colormap: PET_COLORMAP, opacity: FUSION_OPACITY }
    ]
  }, [ctVolumeUrl, predPetVolumeUrl])

  const hasRealPet = Boolean(activeResult?.has_real_pet && realPetVolumeUrl)
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

  const metricsTone = activeResult?.metrics?.evaluation_status === 'failed' ? 'warning' : 'success'
  const processingHeadline = uploading
    ? t('processingRunningHeadline')
    : uploadError
      ? t('processingFailedHeadline')
      : activeResult
        ? t('processingCompletedHeadline')
        : hasPayloadSelected
          ? t('processingPayloadReadyHeadline')
          : t('processingIdleHeadline')
  const processingMessage = uploading
    ? t('processingRunningMessage')
    : uploadError
      ? t('processingFailedMessage')
      : activeResult
        ? t('processingCompletedMessage')
        : hasPayloadSelected
          ? t('processingPayloadReadyMessage')
          : t('processingIdleMessage')
  const processingTone = uploadError ? 'error' : activeResult ? 'success' : uploading || hasPayloadSelected ? 'info' : 'warning'
  const processingStepsCurrent = activeResult ? 2 : stageStep >= 0 ? stageStep : 0
  const processingStepsStatus: 'wait' | 'process' | 'finish' | 'error' = uploadError
    ? 'error'
    : activeResult
      ? 'finish'
      : stageStep >= 0 || hasPayloadSelected
        ? 'process'
        : 'wait'

  const ctSyncPeers = useMemo(() => {
    // 中文说明：三联阅片同步采用“每个 viewer 广播到其它 viewer”的方式。
    // CT viewer 的 peers：Pred（融合）+ Real（如果存在）。
    const peers = [predViewer, realViewer]
    return peers.filter((peer): peer is NiivueViewerLike => Boolean(peer))
  }, [predViewer, realViewer])

  const predSyncPeers = useMemo(() => {
    // Pred viewer 的 peers：CT + Real。
    const peers = [ctViewer, realViewer]
    return peers.filter((peer): peer is NiivueViewerLike => Boolean(peer))
  }, [ctViewer, realViewer])

  const realSyncPeers = useMemo(() => {
    // Real viewer 的 peers：CT + Pred。
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



  const validateNiftiFile = (file: File) => {
    const lower = file.name.toLowerCase()
    const isNifti = lower.endsWith('.nii') || lower.endsWith('.nii.gz')
    if (!isNifti) {
      return t('niftiOnly')
    }
    if (!fileSizeOk(file)) {
      return t('fileMax200mb')
    }
    return null
  }

  const validateDicomDirectoryFiles = (files: File[]) => {
    const tooLarge = files.find((file) => !fileSizeOk(file))
    if (tooLarge) {
      return t('eachDirFileMax200mb')
    }
    return null
  }

  const toBrowserFiles = (fileList: UploadFile[]): File[] => {
    return fileList.reduce<File[]>((files, item) => {
      if (item.originFileObj) {
        files.push(item.originFileObj as File)
      }
      return files
    }, [])
  }

  const normalizeSingleUploadList = (fileList: UploadFile[]) => {
    if (fileList.length === 0) {
      return []
    }
    return [fileList[fileList.length - 1]]
  }

  const handleNiftiCtUploadChange: UploadProps['onChange'] = ({ fileList }) => {
    const nextList = normalizeSingleUploadList(fileList)
    const file = toBrowserFiles(nextList)[0] ?? null
    if (!file) {
      setNiftiCtFile(null)
      setNiftiCtUploadList([])
      return
    }
    const error = validateNiftiFile(file)
    if (error) {
      setUploadError(error)
      setNiftiCtFile(null)
      setNiftiCtUploadList([])
      return
    }
    setUploadError(null)
    setNiftiCtFile(file)
    setNiftiCtUploadList(nextList)
  }

  const handleRealPetUploadChange: UploadProps['onChange'] = ({ fileList }) => {
    const nextList = normalizeSingleUploadList(fileList)
    const file = toBrowserFiles(nextList)[0] ?? null
    if (!file) {
      setRealPetFile(null)
      setRealPetUploadList([])
      return
    }
    const error = validateNiftiFile(file)
    if (error) {
      setUploadError(error)
      setRealPetFile(null)
      setRealPetUploadList([])
      return
    }
    setUploadError(null)
    setRealPetFile(file)
    setRealPetUploadList(nextList)
  }

  const handleDicomDirUploadChange: UploadProps['onChange'] = ({ fileList }) => {
    const files = toBrowserFiles(fileList)
    if (files.length === 0) {
      setDicomDirFiles([])
      setDicomDirUploadList([])
      return
    }
    const error = validateDicomDirectoryFiles(files)
    if (error) {
      setUploadError(error)
      setDicomDirFiles([])
      setDicomDirUploadList([])
      return
    }
    setUploadError(null)
    setDicomDirFiles(files)
    setDicomDirUploadList(fileList)
  }

  const handleRealPetDicomDirUploadChange: UploadProps['onChange'] = ({ fileList }) => {
    const files = toBrowserFiles(fileList)
    if (files.length === 0) {
      setRealPetDicomDirFiles([])
      setRealPetDicomDirUploadList([])
      return
    }
    const error = validateDicomDirectoryFiles(files)
    if (error) {
      setUploadError(error)
      setRealPetDicomDirFiles([])
      setRealPetDicomDirUploadList([])
      return
    }
    setUploadError(null)
    setRealPetDicomDirFiles(files)
    setRealPetDicomDirUploadList(fileList)
  }

  const handleUploadAndInfer = async () => {
    if (uploadMode === 'nifti' && !niftiCtFile) {
      setUploadError(t('uploadCtRequired'))
      return
    }
    if (uploadMode === 'dicom_dir' && dicomDirFiles.length === 0) {
      setUploadError(t('uploadDirRequired'))
      return
    }

    setUploading(true)
    setUploadStage('uploading')
    setUploadError(null)
    setStudyResult(null)
    // no-op: slice index is controlled by Niivue interactions

    try {
      const formData = new FormData()

      // 中文说明：根据上传模式构造 multipart：
      // - nifti: ct_file + 可选 real_pet_file
      // - dicom_dir: dicom_files(多文件) + 可选 real_pet_dicom_files(多文件，PT)
      if (uploadMode === 'nifti' && niftiCtFile) {
        formData.append('ct_file', niftiCtFile)
        if (realPetFile) {
          formData.append('real_pet_file', realPetFile)
        }
      }
      if (uploadMode === 'dicom_dir') {
        dicomDirFiles.forEach((fileObj) => {
          const relativePath = (fileObj as File & { webkitRelativePath?: string }).webkitRelativePath
          formData.append('dicom_files', fileObj, relativePath || fileObj.name)
        })
        realPetDicomDirFiles.forEach((fileObj) => {
          const relativePath = (fileObj as File & { webkitRelativePath?: string }).webkitRelativePath
          formData.append('real_pet_dicom_files', fileObj, relativePath || fileObj.name)
        })
      }

      setUploadStage('processing')
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const uploadMeta = response.data as CaseMeta
      setUploadStage('rendering')

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
    setDicomDirFiles([])
    setRealPetDicomDirFiles([])
    setNiftiCtUploadList([])
    setRealPetUploadList([])
    setDicomDirUploadList([])
    setRealPetDicomDirUploadList([])
    setUploadError(null)
    setUploadStage('idle')
    setStudyResult(null)
    // no-op: slice/colormap/opacity controls removed
  }

  return (
    <div className="medical-dashboard-grid">
      <aside className="medical-sidebar-stack">
        <section className="medical-panel">
          <div className="medical-panel-toolbar">
            <h2 className="medical-panel-title">{t('panelUploadTitle')}</h2>
            <span className={`medical-status-pill medical-status-${backendStatus ? 'online' : 'offline'}`}>
              {backendStatus === null ? t('backendChecking') : backendStatus ? t('backendOnline') : t('backendOffline')}
            </span>
          </div>

          <div className="medical-inline-cluster medical-inline-cluster-compact">
            <span className="medical-section-label">{t('backendStatus')}</span>
            <button type="button" className="medical-button medical-button-ghost" onClick={checkBackendStatus}>
              {t('refresh')}
            </button>
          </div>

          <div className="medical-panel-divider" />

          <p className="medical-section-label">{t('submissionMode')}</p>

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
              <span>{t('modeNifti')}</span>
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
              <span>{t('modeDirDicom')}</span>
            </label>
          </fieldset>

          {uploadMode === 'nifti' && (
            <>
              <label className="medical-file-control">
                <span>{t('ctNiftiRequired')}</span>
                <div data-testid="ct-nifti-input">
                  <Upload
                    fileList={niftiCtUploadList}
                    maxCount={1}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={handleNiftiCtUploadChange}
                    onRemove={() => {
                      setNiftiCtFile(null)
                      setNiftiCtUploadList([])
                      return true
                    }}
                    accept=".nii,.nii.gz"
                  >
                    <button type="button" className="medical-button medical-button-ghost medical-upload-trigger">
                      {t('chooseFile')}
                    </button>
                  </Upload>
                </div>
                <small>{niftiCtFile ? niftiCtFile.name : t('noFileSelected')}</small>
              </label>

              <label className="medical-file-control">
                <span>{t('realPetOptional')}</span>
                <div data-testid="real-pet-input">
                  <Upload
                    fileList={realPetUploadList}
                    maxCount={1}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={handleRealPetUploadChange}
                    onRemove={() => {
                      setRealPetFile(null)
                      setRealPetUploadList([])
                      return true
                    }}
                    accept=".nii,.nii.gz"
                  >
                    <button type="button" className="medical-button medical-button-ghost medical-upload-trigger">
                      {t('chooseFile')}
                    </button>
                  </Upload>
                </div>
                <small>{realPetFile ? realPetFile.name : t('noFileSelected')}</small>
              </label>
            </>
          )}

          {uploadMode === 'dicom_dir' && (
            <>
              <label className="medical-file-control">
                <span>{t('dicomDirRequired')}</span>
                <div data-testid="dicom-dir-input">
                  <Upload
                    fileList={dicomDirUploadList}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={handleDicomDirUploadChange}
                    onRemove={(file) => {
                      const nextList = dicomDirUploadList.filter((item) => item.uid !== file.uid)
                      setDicomDirUploadList(nextList)
                      setDicomDirFiles(toBrowserFiles(nextList))
                      return true
                    }}
                    directory
                    multiple
                  >
                    <button type="button" className="medical-button medical-button-ghost medical-upload-trigger">
                      {t('chooseFolder')}
                    </button>
                  </Upload>
                </div>
                <small>
                  {dicomDirFiles.length > 0 ? t('dicomFilesReady', dicomDirFiles.length) : t('chooseDicomFolder')}
                </small>
              </label>

              <label className="medical-file-control">
                <span>{t('realPetDicomOptional')}</span>
                <div data-testid="real-pet-dicom-dir-input">
                  <Upload
                    fileList={realPetDicomDirUploadList}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={handleRealPetDicomDirUploadChange}
                    onRemove={(file) => {
                      const nextList = realPetDicomDirUploadList.filter((item) => item.uid !== file.uid)
                      setRealPetDicomDirUploadList(nextList)
                      setRealPetDicomDirFiles(toBrowserFiles(nextList))
                      return true
                    }}
                    directory
                    multiple
                  >
                    <button type="button" className="medical-button medical-button-ghost medical-upload-trigger">
                      {t('chooseFolder')}
                    </button>
                  </Upload>
                </div>
                <small>
                  {realPetDicomDirFiles.length > 0
                    ? t('dicomFilesReady', realPetDicomDirFiles.length)
                    : t('chooseDicomFolder')}
                </small>
              </label>
            </>
          )}

          {uploadError && (
            <div className="medical-banner medical-banner-error medical-banner-spaced">
              <div className="medical-banner-title">{t('uploadFailed')}</div>
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
              {uploading ? t('running') : t('runInference')}
            </button>
            <button
              type="button"
              className="medical-button medical-button-secondary"
              onClick={handleReset}
              disabled={uploading}
            >
              {t('reset')}
            </button>
          </div>
        </section>
      </aside>

      <section className="medical-main-stack">
        <section className="medical-panel medical-processing-panel">
          <div className="medical-processing-header">
            <h2 className="medical-panel-title">{t('processingState')}</h2>
            <span className={`medical-processing-pill medical-processing-pill-${processingTone}`}>{processingHeadline}</span>
          </div>
          <p className="medical-processing-summary">{processingMessage}</p>
          <Steps
            className="medical-steps-component"
            current={processingStepsCurrent}
            status={processingStepsStatus}
            items={[
              { title: t('stepUploadPayload'), icon: <CloudUploadOutlined /> },
              { title: t('stepRunInference'), icon: <ExperimentOutlined /> },
              { title: t('stepPrepareResults'), icon: <CheckCircleOutlined /> }
            ]}
          />
        </section>

        <section className="medical-panel medical-workspace-panel">
          <div className="medical-workspace-header">
            <h2 className="medical-panel-title">{t('workspaceTitle')}</h2>
            {activeResult && (
              <div className="medical-workspace-header-meta">
                {t('workspaceMeta', activeResult.num_slices, activeResult.shape.join(' x '))}
              </div>
            )}
          </div>

          {uploadError && (
            <div className="medical-banner medical-banner-error medical-banner-bottom-gap">
              <div className="medical-banner-title">{t('processingFailed')}</div>
              <div>{t('processingFailedDetail')}</div>
            </div>
          )}

          {activeResult && !uploadError && (
            <div
              className={`medical-banner ${
                metricsTone === 'warning' ? 'medical-banner-warning' : 'medical-banner-success'
              } medical-banner-bottom-gap`}
            >
              <div className="medical-banner-title">{t('workspaceReady')}</div>
              <div>{t('workspaceReadyDetail')}</div>
            </div>
          )}

          {!activeResult && (
            <div className="medical-workspace-empty-state">
              {uploading ? (
                <div className="medical-loading-block" aria-label={t('loadingWorkspace')}>
                  <div className="medical-loading-line" />
                  <div className="medical-loading-line" />
                  <div className="medical-loading-line" />
                </div>
              ) : (
                  <div className="medical-empty-state">{t('emptyWorkspace')}</div>
                )}
              </div>
            )}

           {activeResult && (
             <div>
              <div className={`medical-viewer-grid ${hasRealPet ? 'medical-viewer-grid-real' : ''}`}>
                <article className="medical-subpanel">
                  <h3 className="medical-subpanel-title">{t('ctVolume')}</h3>
                  <div className="medical-viewer-frame">
                    <NiivueVolumeViewer
                      volumes={ctVolumes}
                      syncPeers={ctSyncPeers}
                      onViewerReady={handleCtViewerReady}
                    />
                  </div>
                </article>

                {hasRealPet && (
                  <article className="medical-subpanel">
                    <h3 className="medical-subpanel-title">{t('realPetReference')}</h3>
                    <div className="medical-viewer-frame">
                      <NiivueVolumeViewer
                        volumes={realPetVolumes}
                        syncPeers={realSyncPeers}
                        onViewerReady={handleRealViewerReady}
                      />
                    </div>
                  </article>
                )}

                <article className="medical-subpanel">
                  <h3 className="medical-subpanel-title">{t('predictedPetFusion')}</h3>
                  <div className="medical-viewer-frame">
                    <NiivueVolumeViewer
                      volumes={predictedFusionVolumes}
                      syncPeers={predSyncPeers}
                      onViewerReady={handlePredViewerReady}
                    />
                  </div>
                </article>
              </div>
            </div>
          )}
        </section>

        {hasMetrics && activeResult && (
          <section className="medical-panel">
            <h2 className="medical-panel-title">{t('metrics')}</h2>
            <div className="medical-metrics-grid">
              <div className="medical-metric-item">
                <div className="medical-metric-label">{t('inferenceMs')}</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.inference_time_ms, 1, t('metricNotAvailable'))}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">{t('slices')}</div>
                <div className="medical-metric-value">{activeResult.metrics.slices_processed ?? t('metricNotAvailable')}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">PSNR</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.psnr, 3, t('metricNotAvailable'))}</div>
              </div>
              <div className="medical-metric-item">
                <div className="medical-metric-label">SSIM</div>
                <div className="medical-metric-value">{formatMetric(activeResult.metrics.ssim, 4, t('metricNotAvailable'))}</div>
              </div>
            </div>
            {activeResult.metrics.evaluation_status && (
              <div
                className={`medical-banner ${
                  activeResult.metrics.evaluation_status === 'failed'
                    ? 'medical-banner-error'
                    : 'medical-banner-info'
                } medical-banner-tight`}
              >
                <div>
                  {t(
                    'evaluation',
                    activeResult.metrics.evaluation_status,
                    activeResult.metrics.evaluation_reason
                  )}
                </div>
              </div>
            )}
          </section>
        )}
      </section>
    </div>
  )
}

export default CTUpload
