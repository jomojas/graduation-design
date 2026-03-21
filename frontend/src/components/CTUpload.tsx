import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Empty,
  Row,
  Slider,
  Spin,
  Tag,
  Upload,
  message
} from 'antd'
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  CloudUploadOutlined,
  SyncOutlined
} from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import axios from 'axios'

const API_BASE = 'http://localhost:8000'

type CaseMeta = {
  job_id: string
  num_slices: number
  shape: [number, number, number]
  has_real_pet: boolean
}

const CTUpload: React.FC = () => {
  const [ctFileList, setCtFileList] = useState<UploadFile[]>([])
  const [realPetFileList, setRealPetFileList] = useState<UploadFile[]>([])
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null)
  const [uploading, setUploading] = useState(false)
  const [caseMeta, setCaseMeta] = useState<CaseMeta | null>(null)
  const [sliceIndex, setSliceIndex] = useState(0)

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

  const validateNifti = (file: File) => {
    const lower = file.name.toLowerCase()
    const isNifti = lower.endsWith('.nii') || lower.endsWith('.nii.gz')
    if (!isNifti) {
      message.error('Only .nii or .nii.gz files are supported')
      return Upload.LIST_IGNORE
    }
    const isLt200M = file.size / 1024 / 1024 <= 200
    if (!isLt200M) {
      message.error('File size must be <= 200MB')
      return Upload.LIST_IGNORE
    }
    return false
  }

  const sliceImageUrls = useMemo(() => {
    if (!caseMeta) {
      return null
    }
    const ts = Date.now()
    const base = `${API_BASE}/cases/${caseMeta.job_id}/slice/${sliceIndex}`
    return {
      ct: `${base}?view=ct&t=${ts}`,
      realPet: `${base}?view=real_pet&t=${ts}`,
      predPet: `${base}?view=pred_pet&t=${ts}`
    }
  }, [caseMeta, sliceIndex])

  const handleUploadAndInfer = async () => {
    const ct = ctFileList[0]?.originFileObj
    if (!ct) {
      message.error('Please upload a CT NIfTI file')
      return
    }

    setUploading(true)
    setCaseMeta(null)
    setSliceIndex(0)

    try {
      const formData = new FormData()
      formData.append('ct_file', ct)

      const realPet = realPetFileList[0]?.originFileObj
      if (realPet) {
        formData.append('real_pet_file', realPet)
      }

      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setCaseMeta(response.data)
      message.success('Inference complete')
    } catch (error: any) {
      const errorMessage = error?.response?.data?.detail || 'Upload or inference failed'
      message.error(errorMessage)
    } finally {
      setUploading(false)
    }
  }

  const handleReset = () => {
    setCtFileList([])
    setRealPetFileList([])
    setCaseMeta(null)
    setSliceIndex(0)
  }

  return (
    <div>
      <Row gutter={[20, 20]}>
        <Col xs={24} xl={8}>
          <Card title="Backend Status" style={{ marginBottom: 20 }}>
            <Tag
              icon={backendStatus ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
              color={backendStatus ? 'success' : 'error'}
            >
              {backendStatus === null ? 'Checking...' : backendStatus ? 'Online' : 'Offline'}
            </Tag>
            <Button type="link" icon={<SyncOutlined />} onClick={checkBackendStatus}>
              Refresh
            </Button>
          </Card>

          <Card title="Upload Volumes" style={{ marginBottom: 20 }}>
            <p style={{ marginBottom: 8 }}>CT NIfTI (required)</p>
            <Upload
              fileList={ctFileList}
              beforeUpload={validateNifti}
              onChange={(info) => setCtFileList(info.fileList.slice(-1))}
              accept=".nii,.nii.gz"
              maxCount={1}
            >
              <Button icon={<CloudUploadOutlined />}>Select CT</Button>
            </Upload>

            <p style={{ marginTop: 16, marginBottom: 8 }}>Real PET NIfTI (optional)</p>
            <Upload
              fileList={realPetFileList}
              beforeUpload={validateNifti}
              onChange={(info) => setRealPetFileList(info.fileList.slice(-1))}
              accept=".nii,.nii.gz"
              maxCount={1}
            >
              <Button icon={<CloudUploadOutlined />}>Select Real PET</Button>
            </Upload>

            <div style={{ marginTop: 20, display: 'flex', gap: 8 }}>
              <Button
                type="primary"
                onClick={handleUploadAndInfer}
                loading={uploading}
                disabled={ctFileList.length === 0}
              >
                Run 2.5D Inference
              </Button>
              <Button onClick={handleReset} disabled={uploading}>
                Reset
              </Button>
            </div>
          </Card>

          <Card title="Mode Notes">
            <Alert
              type="info"
              showIcon
              message="Evaluation Mode"
              description="Upload CT + Real PET to compare real and predicted PET side-by-side."
            />
            <div style={{ marginTop: 12, color: '#5b6473' }}>
              Model uses 7-slice sliding window with edge replication padding.
            </div>
          </Card>
        </Col>

        <Col xs={24} xl={16}>
          <Card
            title="Synchronized Triple View"
            extra={
              caseMeta ? `Slices: ${caseMeta.num_slices} | Shape: ${caseMeta.shape.join(' x ')}` : ''
            }
          >
            {!caseMeta && (
              <div style={{ minHeight: 420, display: 'grid', placeItems: 'center' }}>
                <Empty description="Upload CT volume to start" />
              </div>
            )}

            {caseMeta && (
              <Spin spinning={uploading}>
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={8}>
                    <Card size="small" title="Original CT">
                      {sliceImageUrls && (
                        <img src={sliceImageUrls.ct} alt="CT" style={{ width: '100%', borderRadius: 8 }} />
                      )}
                    </Card>
                  </Col>
                  <Col xs={24} md={8}>
                    <Card size="small" title="Real PET (Hot)">
                      {caseMeta.has_real_pet && sliceImageUrls ? (
                        <img
                          src={sliceImageUrls.realPet}
                          alt="Real PET"
                          style={{ width: '100%', borderRadius: 8 }}
                        />
                      ) : (
                        <Empty description="Not provided" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                      )}
                    </Card>
                  </Col>
                  <Col xs={24} md={8}>
                    <Card size="small" title="Predicted PET (Hot)">
                      {sliceImageUrls && (
                        <img
                          src={sliceImageUrls.predPet}
                          alt="Predicted PET"
                          style={{ width: '100%', borderRadius: 8 }}
                        />
                      )}
                    </Card>
                  </Col>
                </Row>

                <div style={{ marginTop: 20 }}>
                  <div style={{ marginBottom: 8, color: '#596273' }}>
                    Slice index: {sliceIndex}
                  </div>
                  <Slider
                    min={0}
                    max={Math.max(caseMeta.num_slices - 1, 0)}
                    value={sliceIndex}
                    onChange={(value) => setSliceIndex(Array.isArray(value) ? value[0] : value)}
                  />
                </div>
              </Spin>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default CTUpload
