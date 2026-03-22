import React from 'react'

export function DashboardShell({ children }: { children?: React.ReactNode }) {
  return (
    <div className="medical-console-layout">
      <header className="medical-console-header">
        <h1 className="medical-console-title">CT to PET Synthesis Workbench</h1>
        <p className="medical-console-subtitle">
          2.5D inference with synchronized CT / Real PET / Pred PET review
        </p>
        <p className="medical-console-caption">Research prototype for non-diagnostic use</p>
      </header>
      <main>{children}</main>
    </div>
  )
}
