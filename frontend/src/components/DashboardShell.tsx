import React from 'react'

export function DashboardShell({ children }: { children?: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-medical-bg text-medical-text">
      <header className="border-b border-medical-surface px-6 py-4">
        <h1 className="text-2xl font-bold">CT to PET Medical Imaging Platform</h1>
        <p className="text-sm text-medical-muted">Research Prototype - Non-Diagnostic Use Only</p>
      </header>
      <div className="flex">
        <aside className="w-64 border-r border-medical-surface p-4">
          {/* Sidebar slot for metadata cards */}
          <div className="text-medical-muted text-sm">Metadata Panel</div>
        </aside>
        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
