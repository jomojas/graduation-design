import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Mocked } from 'vitest'
import { render, screen } from '@testing-library/react'
import axios from 'axios'
import type { AxiosStatic } from 'axios'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
  },
}))

import App from './App'

const mockedAxios = axios as Mocked<AxiosStatic>

beforeEach(() => {
  mockedAxios.get.mockResolvedValue({})
})

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />)
    // Check that the header title is rendered
    const titleElement = screen.getByText('CT to PET Synthesis Workbench')
    expect(titleElement).toBeInTheDocument()
  })

  it('displays the subtitle in header', () => {
    render(<App />)
    const subtitleElement = screen.getByText('2.5D inference with synchronized CT / Real PET / Pred PET review')
    expect(subtitleElement).toBeInTheDocument()
  })

  it('renders core workspace sections', () => {
    render(<App />)
    expect(screen.getByText('Upload Volumes')).toBeInTheDocument()
    expect(screen.getByText('Synchronized Imaging Workspace')).toBeInTheDocument()
  })
})
