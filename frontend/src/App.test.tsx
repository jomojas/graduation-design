import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Mocked } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import axios from 'axios'
import type { AxiosStatic } from 'axios'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

import App from './App'
import { LanguageProvider } from './i18n/LanguageProvider'
import { LANGUAGE_STORAGE_KEY } from './i18n'

const mockedAxios = axios as Mocked<AxiosStatic>

beforeEach(() => {
  mockedAxios.get.mockResolvedValue({})
  localStorage.clear()
})

describe('App', () => {
  it('renders without crashing', () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN')
    render(
      <LanguageProvider>
        <App />
      </LanguageProvider>
    )
    // Check that the header title is rendered
    const titleElement = screen.getByText('CT 到 PET 合成工作台')
    expect(titleElement).toBeInTheDocument()
  })

  it('displays the subtitle in header', () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN')
    render(
      <LanguageProvider>
        <App />
      </LanguageProvider>
    )
    const subtitleElement = screen.getByText('2.5D 推理与 CT / 真实 PET / 预测 PET 同步阅片')
    expect(subtitleElement).toBeInTheDocument()
  })

  it('renders core workspace sections', () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN')
    render(
      <LanguageProvider>
        <App />
      </LanguageProvider>
    )
    expect(screen.getByRole('heading', { name: '上传数据' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '同步影像工作区' })).toBeInTheDocument()
  })

  it('switches between Chinese and English labels', async () => {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN')
    render(
      <LanguageProvider>
        <App />
      </LanguageProvider>
    )

    expect(screen.getByText('CT 到 PET 合成工作台')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'EN' }))
    await waitFor(() => {
      expect(screen.getByText('CT to PET Synthesis Workbench')).toBeInTheDocument()
    })
  })
})
