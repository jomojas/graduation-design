import React from 'react'
import { Layout, Typography } from 'antd'
import CTUpload from './components/CTUpload'

const { Header, Content } = Layout
const { Title } = Typography

const App: React.FC = () => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        background: 'linear-gradient(135deg, #1890ff 0%, #40a9ff 100%)', 
        padding: '0 50px',
        height: 'auto',
        lineHeight: 'normal'
      }}>
        <Title level={3} style={{ color: '#fff', margin: '16px 0' }}>
          CT转PET医学影像转换系统
        </Title>
        <p style={{ color: 'rgba(255,255,255,0.9)', margin: '0 0 16px 0' }}>
          基于深度学习的医学影像转换平台 | GAN图像生成技术
        </p>
      </Header>
      <Content style={{ padding: '24px 50px', background: '#f5f5f5' }}>
        <CTUpload />
      </Content>
    </Layout>
  )
}

export default App
