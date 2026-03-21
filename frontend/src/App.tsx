import React from 'react'
import { Layout, Typography } from 'antd'
import CTUpload from './components/CTUpload'

const { Header, Content } = Layout
const { Title } = Typography

const App: React.FC = () => {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ 
        background: 'linear-gradient(120deg, #0f3b66 0%, #1b6ca8 45%, #4fb286 100%)', 
        padding: '0 50px',
        height: 'auto',
        lineHeight: 'normal'
      }}>
        <Title level={3} style={{ color: '#fff', margin: '16px 0' }}>
          CT to PET Synthesis Workbench
        </Title>
        <p style={{ color: 'rgba(255,255,255,0.9)', margin: '0 0 16px 0' }}>
          2.5D inference with synchronized CT / Real PET / Pred PET review
        </p>
      </Header>
      <Content style={{ padding: '24px 50px', background: '#f5f5f5' }}>
        <CTUpload />
      </Content>
    </Layout>
  )
}

export default App
