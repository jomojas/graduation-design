import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Mocked } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import axios from 'axios'
import type { AxiosStatic } from 'axios'

vi.mock('./NiivueVolumeViewer', () => ({
  default: ({ volumes }: { volumes?: Array<{ url: string }> }) => (
    <div data-testid="mock-niivue-viewer">{volumes?.map((volume) => volume.url).join('|')}</div>
  ),
}))

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

import CTUpload from './CTUpload'
import { LanguageProvider } from '../i18n/LanguageProvider'
import { LANGUAGE_STORAGE_KEY } from '../i18n'

const mockedAxios = axios as Mocked<AxiosStatic>

const renderUpload = () => {
  return render(
    <LanguageProvider>
      <CTUpload />
    </LanguageProvider>
  )
}

const getUploadInput = (testId: string) => {
  const wrapper = screen.getByTestId(testId)
  const input = wrapper.querySelector('input[type="file"]')
  if (!input) {
    throw new Error(`Upload input not found for ${testId}`)
  }
  return input as HTMLInputElement
}

const mockCaseMeta = {
  job_id: 'job-123',
  study_id: 'study-123',
  num_slices: 3,
  shape: [128, 128, 64],
  has_real_pet: false,
}

const mockCaseMetaWithPet = {
  job_id: 'job-456',
  study_id: 'study-456',
  num_slices: 4,
  shape: [64, 64, 32],
  has_real_pet: true,
}

const buildResultResponse = (data: typeof mockCaseMeta | typeof mockCaseMetaWithPet) => ({
  success: true,
  study_id: data.study_id,
  job_id: data.job_id,
  has_real_pet: data.has_real_pet,
  num_slices: data.num_slices,
  shape: data.shape,
  metrics: {
    inference_time_ms: 123.4,
    output_shape: data.shape,
    slices_processed: data.num_slices,
    psnr: data.has_real_pet ? 24.12 : null,
    ssim: data.has_real_pet ? 0.8123 : null,
    evaluation_status: data.has_real_pet ? 'completed' : 'skipped',
    evaluation_reason: data.has_real_pet ? null : 'reference_missing',
  },
  ct: {
    available: true,
    nifti_path: `/uploads/${data.job_id}/ct.nii.gz`,
    slice_endpoint_template: `/cases/${data.job_id}/slice/{index}?view=ct`,
  },
  predicted_pet: {
    available: true,
    nifti_path: `/outputs/${data.job_id}/pred_pet.nii.gz`,
    slice_endpoint_template: `/cases/${data.job_id}/slice/{index}?view=pred_pet`,
  },
  real_pet: {
    available: data.has_real_pet,
    nifti_path: data.has_real_pet ? `/uploads/${data.job_id}/real_pet.nii.gz` : null,
    slice_endpoint_template: data.has_real_pet
      ? `/cases/${data.job_id}/slice/{index}?view=real_pet`
      : null,
  },
})

beforeEach(() => {
  localStorage.clear()
  localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN')
  mockedAxios.get.mockClear()
  mockedAxios.post.mockClear()
  mockedAxios.get.mockImplementation((url?: string) => {
    if (typeof url === 'string' && url.includes('/studies/')) {
      return Promise.resolve({ data: buildResultResponse(mockCaseMeta) })
    }
    return Promise.resolve({})
  })
  mockedAxios.post.mockResolvedValue({ data: mockCaseMeta })
})

describe('CTUpload', () => {
  it('shows empty workspace and disabled run button by default', () => {
    renderUpload()
    expect(screen.getByText('请先提交一组数据以开始')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'NIfTI' })).toBeChecked()
    expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeDisabled()
  })

  it('submits NIfTI studies and renders case metadata', async () => {
    renderUpload()
    const ctInput = getUploadInput('ct-nifti-input')
    const ctFile = new File(['dummy'], 'scan.nii', { type: 'application/octet-stream' })
    fireEvent.change(ctInput, { target: { files: [ctFile] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalled()
    })

    const payload = mockedAxios.post.mock.calls[0]?.[1] as FormData
    expect(payload.get('ct_file')).toBe(ctFile)

    await waitFor(() => {
      expect(screen.getByText('切片数: 3 | 形状: 128 x 128 x 64')).toBeInTheDocument()
    })

    expect(screen.getByText('CT 体数据')).toBeInTheDocument()
    expect(screen.getByText('预测 PET 融合图')).toBeInTheDocument()
    expect(screen.queryByText('真实 PET 参考')).not.toBeInTheDocument()
    expect(screen.getByText('指标')).toBeInTheDocument()
    expect(screen.getByText('评估结果: skipped (reference_missing)')).toBeInTheDocument()
    expect(screen.getAllByText('无')).toHaveLength(2)
    expect(screen.queryByText('0.000')).not.toBeInTheDocument()
    expect(screen.queryByText('0.0000')).not.toBeInTheDocument()
  })

  it('submits browser directory DICOM as repeated dicom_files', async () => {
    renderUpload()
    fireEvent.click(screen.getByRole('radio', { name: '目录 DICOM' }))

    const directoryInput = getUploadInput('dicom-dir-input')
    const dicom1 = new File(['a'], '1.dcm', { type: 'application/dicom' })
    const dicom2 = new File(['b'], '2.dcm', { type: 'application/dicom' })
    Object.defineProperty(dicom1, 'webkitRelativePath', { value: 'study/1.dcm' })
    Object.defineProperty(dicom2, 'webkitRelativePath', { value: 'study/2.dcm' })
    fireEvent.change(directoryInput, { target: { files: [dicom1, dicom2] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalled()
    })

    const payload = mockedAxios.post.mock.calls[0]?.[1] as FormData
    const uploadedDicomFiles = payload.getAll('dicom_files') as File[]
    expect(uploadedDicomFiles.length).toBe(2)
    expect(uploadedDicomFiles[0]?.name).toBe('study/1.dcm')
    expect(uploadedDicomFiles[1]?.name).toBe('study/2.dcm')
  })

  it('submits browser directory DICOM with optional real PET DICOM folder payload', async () => {
    renderUpload()
    fireEvent.click(screen.getByRole('radio', { name: '目录 DICOM' }))

    const directoryInput = getUploadInput('dicom-dir-input')
    const dicom1 = new File(['a'], '1.dcm', { type: 'application/dicom' })
    const dicom2 = new File(['b'], '2.dcm', { type: 'application/dicom' })
    Object.defineProperty(dicom1, 'webkitRelativePath', { value: 'study/1.dcm' })
    Object.defineProperty(dicom2, 'webkitRelativePath', { value: 'study/2.dcm' })
    fireEvent.change(directoryInput, { target: { files: [dicom1, dicom2] } })

    const petDirInput = getUploadInput('real-pet-dicom-dir-input')
    const pet1 = new File(['p1'], 'p1.dcm', { type: 'application/dicom' })
    const pet2 = new File(['p2'], 'p2.dcm', { type: 'application/dicom' })
    Object.defineProperty(pet1, 'webkitRelativePath', { value: 'pet/1.dcm' })
    Object.defineProperty(pet2, 'webkitRelativePath', { value: 'pet/2.dcm' })
    fireEvent.change(petDirInput, { target: { files: [pet1, pet2] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalled()
    })

    const payload = mockedAxios.post.mock.calls[0]?.[1] as FormData
    const uploadedDicomFiles = payload.getAll('dicom_files') as File[]
    expect(uploadedDicomFiles.length).toBe(2)
    const uploadedPetFiles = payload.getAll('real_pet_dicom_files') as File[]
    expect(uploadedPetFiles.length).toBe(2)
    expect(payload.get('real_pet_file')).toBeNull()
  })

  it('renders inline backend validation detail on upload error', async () => {
    mockedAxios.post.mockRejectedValueOnce({
      response: {
        data: {
          detail: 'DICOM ZIP mode has been removed',
        },
      },
    })
    renderUpload()

    fireEvent.click(screen.getByRole('radio', { name: '目录 DICOM' }))

    const directoryInput = getUploadInput('dicom-dir-input')
    const dicom1 = new File(['a'], '1.dcm', { type: 'application/dicom' })
    Object.defineProperty(dicom1, 'webkitRelativePath', { value: 'study/1.dcm' })
    fireEvent.change(directoryInput, { target: { files: [dicom1] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    expect(await screen.findByText('DICOM ZIP mode has been removed')).toBeInTheDocument()
  })

  it('shows staged processing steps while request is running', async () => {
    let resolvePost!: (value: { data: typeof mockCaseMetaWithPet }) => void
    mockedAxios.post.mockImplementationOnce(
      () =>
        new Promise<{ data: typeof mockCaseMetaWithPet }>((resolve) => {
          resolvePost = resolve
        })
    )

    mockedAxios.get.mockImplementation((url?: string) => {
      if (typeof url === 'string' && url.includes('/studies/')) {
        return Promise.resolve({ data: buildResultResponse(mockCaseMetaWithPet) })
      }
      return Promise.resolve({})
    })

    renderUpload()
    const ctInput = getUploadInput('ct-nifti-input')
    const ctFile = new File(['dummy'], 'scan.nii', { type: 'application/octet-stream' })
    fireEvent.change(ctInput, { target: { files: [ctFile] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))
    expect(screen.getAllByText('上传数据').length).toBeGreaterThan(0)
    expect(screen.getByText('执行推理')).toBeInTheDocument()
    expect(screen.getByText('准备结果')).toBeInTheDocument()

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalled()
    })

    resolvePost({ data: mockCaseMetaWithPet })

    await waitFor(() => {
      expect(screen.getByText('切片数: 4 | 形状: 64 x 64 x 32')).toBeInTheDocument()
    })

    expect(screen.getByText('真实 PET 参考')).toBeInTheDocument()
  })

  it('hides metrics card when all metrics values are missing', async () => {
    mockedAxios.get.mockImplementation((url?: string) => {
      if (typeof url === 'string' && url.includes('/studies/')) {
        return Promise.resolve({
          data: {
            ...buildResultResponse(mockCaseMeta),
            metrics: {
              inference_time_ms: null,
              output_shape: null,
              slices_processed: null,
              psnr: null,
              ssim: null,
              evaluation_status: null,
              evaluation_reason: null,
            },
          },
        })
      }
      return Promise.resolve({})
    })

    renderUpload()
    const ctInput = getUploadInput('ct-nifti-input')
    const ctFile = new File(['dummy'], 'scan.nii', { type: 'application/octet-stream' })
    fireEvent.change(ctInput, { target: { files: [ctFile] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    await waitFor(() => {
      expect(screen.getByText('预测 PET 融合图')).toBeInTheDocument()
    })

    expect(screen.queryByText('指标')).not.toBeInTheDocument()
  })

  it('renders workspace from nifti_path without legacy slice templates', async () => {
    mockedAxios.get.mockImplementation((url?: string) => {
      if (typeof url === 'string' && url.includes('/studies/')) {
        const result = buildResultResponse(mockCaseMeta)
        return Promise.resolve({
          data: {
            ...result,
            ct: {
              ...result.ct,
              slice_endpoint_template: null,
            },
            predicted_pet: {
              ...result.predicted_pet,
              slice_endpoint_template: null,
            },
            real_pet: {
              ...result.real_pet,
              slice_endpoint_template: null,
            },
          },
        })
      }
      return Promise.resolve({})
    })

    renderUpload()
    const ctInput = getUploadInput('ct-nifti-input')
    const ctFile = new File(['dummy'], 'scan.nii', { type: 'application/octet-stream' })
    fireEvent.change(ctInput, { target: { files: [ctFile] } })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '开始 2.5D 推理' })).toBeEnabled()
    })

    fireEvent.click(screen.getByRole('button', { name: '开始 2.5D 推理' }))

    await waitFor(() => {
      expect(screen.getByText('预测 PET 融合图')).toBeInTheDocument()
    })

    const viewers = screen.getAllByTestId('mock-niivue-viewer')
    expect(viewers[0]?.textContent).toContain('http://localhost:8000/uploads/job-123/ct.nii.gz')
    expect(viewers[1]?.textContent).toContain('http://localhost:8000/outputs/job-123/pred_pet.nii.gz')
  })
})
