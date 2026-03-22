import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, waitFor } from '@testing-library/react'
import NiivueVolumeViewer from './NiivueVolumeViewer'

type MockViewer = {
  attachToCanvas: ReturnType<typeof vi.fn>
  loadVolumes: ReturnType<typeof vi.fn>
  broadcastTo: ReturnType<typeof vi.fn>
  cleanup: ReturnType<typeof vi.fn>
}

const mockViewers: MockViewer[] = []

vi.mock('@niivue/niivue', () => {
  class MockNiivue {
    attachToCanvas = vi.fn()
    loadVolumes = vi.fn().mockResolvedValue(undefined)
    broadcastTo = vi.fn()
    cleanup = vi.fn()

    constructor() {
      mockViewers.push(this)
    }
  }

  return {
    Niivue: MockNiivue,
  }
})

describe('NiivueVolumeViewer', () => {
  beforeEach(() => {
    mockViewers.length = 0
  })

  it('attaches viewer to canvas and loads nifti volumes', async () => {
    render(<NiivueVolumeViewer volumeUrls={['/cases/ct.nii.gz', '/cases/readme.txt']} />)

    await waitFor(() => {
      expect(mockViewers).toHaveLength(1)
      expect(mockViewers[0].attachToCanvas).toHaveBeenCalledTimes(1)
      expect(mockViewers[0].loadVolumes).toHaveBeenCalledWith([{ url: '/cases/ct.nii.gz' }])
    })
  })

  it('calls cleanup on unmount', async () => {
    const { unmount } = render(<NiivueVolumeViewer volumeUrls={['/cases/ct.nii.gz']} />)

    await waitFor(() => {
      expect(mockViewers).toHaveLength(1)
    })

    unmount()

    expect(mockViewers[0].cleanup).toHaveBeenCalledTimes(1)
  })

  it('synchronizes viewer to peers via broadcastTo', async () => {
    const peer = {
      attachToCanvas: vi.fn(),
      loadVolumes: vi.fn(),
      broadcastTo: vi.fn(),
      cleanup: vi.fn(),
    }

    render(<NiivueVolumeViewer volumeUrls={['/cases/pet.nii']} syncPeers={[peer]} />)

    await waitFor(() => {
      expect(mockViewers).toHaveLength(1)
      expect(mockViewers[0].broadcastTo).toHaveBeenCalledWith(peer)
    })
  })
})
