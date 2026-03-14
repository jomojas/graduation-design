import React, { useState } from 'react'
import { Card, Upload, Button, Progress, message, Row, Col, Tag, Spin } from 'antd'
import { UploadOutlined, PictureOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import axios from 'axios'

const CTUpload: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [converting, setConverting] = useState(false)
  const [progress, setProgress] = useState(0)
  const [petImageUrl, setPetImageUrl] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null)
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null)

  React.useEffect(() => {
    checkBackendStatus()
  }, [])

  const checkBackendStatus = async () => {
    try {
      await axios.get('http://localhost:8000/')
      setBackendStatus(true)
    } catch {
      setBackendStatus(false)
    }
  }

  const uploadProps = {
    name: 'file',
    fileList,
    beforeUpload: (file: File) => {
      const isImage = file.type.startsWith('image/')
      if (!isImage) {
        message.error('请上传图片文件！')
        return false
      }
      const isLt10M = file.size / 1024 / 1024 < 10
      if (!isLt10M) {
        message.error('图片大小不能超过10MB！')
        return false
      }
      return false
    },
    onChange: (info: any) => {
      setFileList(info.fileList.slice(-1))
      setPetImageUrl(null)
      if (info.fileList.length > 0 && info.fileList[0].originFileObj) {
        const file = info.fileList[0].originFileObj
        const url = URL.createObjectURL(file)
        setOriginalImageUrl(url)
      }
    },
    accept: '.jpg,.jpeg,.png,.dcm',
    maxCount: 1
  }

  const handleConvert = async () => {
    if (fileList.length === 0) {
      message.error('请先上传CT图片！')
      return
    }

    const file = fileList[0].originFileObj
    if (!file) return

    setConverting(true)
    setProgress(0)
    setPetImageUrl(null)

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return prev
        }
        return prev + 10
      })
    }, 200)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post('http://localhost:8000/convert', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (response.data.success) {
        setPetImageUrl(response.data.pet_image_url)
        message.success('转换成功！')
      } else {
        message.error(response.data.message || '转换失败')
      }
    } catch (error: any) {
      console.error('转换失败:', error)
      message.error(error.response?.data?.message || '转换过程中发生错误')
    } finally {
      setConverting(false)
    }
  }

  const handleReset = () => {
    setFileList([])
    setPetImageUrl(null)
    setProgress(0)
    setOriginalImageUrl(null)
  }

  return (
    <div>
      <Row gutter={[24, 24]}>
        <Col xs={24} lg={12}>
          <Card title="服务状态" style={{ marginBottom: 24 }}>
            <Tag icon={backendStatus ? <CheckCircleOutlined /> : <CloseCircleOutlined />} color={backendStatus ? 'success' : 'error'}>
              {backendStatus === null ? '检查中...' : backendStatus ? '后端服务运行正常' : '后端服务未运行'}
            </Tag>
            <Button type="link" onClick={checkBackendStatus} style={{ marginLeft: 16 }}>刷新状态</Button>
          </Card>

          <Card title="上传CT图片" style={{ marginBottom: 24 }}>
            <Upload {...uploadProps} listType="picture">
              <Button icon={<UploadOutlined />}>选择CT图片</Button>
            </Upload>
            <div style={{ marginTop: 16 }}>
              <Button type="primary" icon={<PictureOutlined />} onClick={handleConvert} loading={converting} disabled={fileList.length === 0} style={{ marginRight: 8 }}>
                开始转换
              </Button>
              <Button onClick={handleReset} disabled={converting || fileList.length === 0}>重置</Button>
            </div>
            {converting && (
              <div style={{ marginTop: 24 }}>
                <Progress percent={progress} status="active" />
                <p style={{ textAlign: 'center', color: '#666', marginTop: 8 }}>正在转换中，请稍候...</p>
              </div>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="使用说明">
            <h4 style={{ marginBottom: 12 }}>支持格式：</h4>
            <Tag>JPG</Tag><Tag>PNG</Tag><Tag>DICOM</Tag>
            <h4 style={{ margin: '16px 0 12px' }}>文件要求：</h4>
            <ul style={{ paddingLeft: 20, color: '#666' }}><li>文件大小不超过10MB</li><li>图片尺寸建议为256x256或512x512</li></ul>
            <h4 style={{ margin: '16px 0 12px' }}>转换流程：</h4>
            <ol style={{ paddingLeft: 20, color: '#666' }}><li>点击"选择CT图片"上传CT图像</li><li>点击"开始转换"按钮</li><li>等待转换完成，查看生成的PET图像</li></ol>
          </Card>
        </Col>
      </Row>
      {petImageUrl && (
        <Card title="转换结果 - PET图像">
          <Row gutter={[24, 24]}>
            <Col xs={24} md={12}>
              <div style={{ textAlign: 'center' }}><h4 style={{ marginBottom: 16 }}>原始CT图像</h4>{originalImageUrl && <img src={originalImageUrl} alt="CT" style={{ maxWidth: '100%', maxHeight: 300, borderRadius: 8 }} />}</div>
            </Col>
            <Col xs={24} md={12}>
              <div style={{ textAlign: 'center' }}><h4 style={{ marginBottom: 16 }}>生成的PET图像</h4><Spin spinning={converting}><img src={petImageUrl} alt="PET" style={{ maxWidth: '100%', maxHeight: 300, borderRadius: 8 }} /></Spin></div>
            </Col>
          </Row>
        </Card>
      )}
    </div>
  )
}

export default CTUpload
