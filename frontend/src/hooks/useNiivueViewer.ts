import { useEffect, useRef, type RefObject } from 'react'
import { Niivue } from '@niivue/niivue'
import { synchronizeNiivuePeers, type NiivueSyncPeer } from '../utils/niivueSync'

const isNiftiVolume = (url: string) => {
  const normalized = url.toLowerCase()
  return normalized.endsWith('.nii') || normalized.endsWith('.nii.gz')
}

type NiivueVolume = {
  url: string
  colormap?: string
  opacity?: number
}

type NiivueViewerLike = NiivueSyncPeer & {
  attachToCanvas: (canvas: HTMLCanvasElement) => void
  loadVolumes: (volumes: NiivueVolume[]) => Promise<void>
  cleanup: () => void
}

type UseNiivueViewerArgs = {
  canvasRef: RefObject<HTMLCanvasElement | null>
  volumes: NiivueVolume[]
  syncPeers?: NiivueViewerLike[]
  onViewerReady?: (viewer: NiivueViewerLike) => void
}

export const useNiivueViewer = ({
  canvasRef,
  volumes,
  syncPeers = [],
  onViewerReady,
}: UseNiivueViewerArgs) => {
  const viewerRef = useRef<NiivueViewerLike | null>(null)

  useEffect(() => {
    const viewer = new Niivue() as unknown as NiivueViewerLike
    viewerRef.current = viewer
    onViewerReady?.(viewer)

    return () => {
      viewer.cleanup()
      if (viewerRef.current === viewer) {
        viewerRef.current = null
      }
    }
  }, [onViewerReady])

  useEffect(() => {
    const canvas = canvasRef.current
    const viewer = viewerRef.current

    if (!canvas || !viewer) {
      return
    }

    viewer.attachToCanvas(canvas)
  }, [canvasRef])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) {
      return
    }

    const resolvedVolumes = volumes.filter((volume) => isNiftiVolume(volume.url))
    if (resolvedVolumes.length === 0) {
      return
    }

    void viewer.loadVolumes(resolvedVolumes)
  }, [volumes])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || syncPeers.length === 0) {
      return
    }

    synchronizeNiivuePeers(viewer, syncPeers)
  }, [syncPeers])

  return viewerRef
}

export type { NiivueViewerLike }
