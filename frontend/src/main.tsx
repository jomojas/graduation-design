import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import { LanguageProvider } from './i18n/LanguageProvider'
import './index.css'
import './styles/globals.css'

// 寻找 index.html 中 id 为 'root' 的元素，并将 React 内容渲染进去
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <LanguageProvider>
      <App />
    </LanguageProvider>
  </React.StrictMode>,
)
