import React from 'react'
import CTUpload from './components/CTUpload'
import { DashboardShell } from './components/DashboardShell'

const App: React.FC = () => {
  return (
    <DashboardShell>
      <div className="medical-console-content">
        <CTUpload />
      </div>
    </DashboardShell>
  )
}

export default App
