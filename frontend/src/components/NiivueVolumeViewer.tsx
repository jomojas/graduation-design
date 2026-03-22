import React, { useRef } from 'react'
import { useNiivueViewer, type NiivueViewerLike } from '../hooks/useNiivueViewer'

type VolumeConfig = {
  url: string
  colormap?: string
  opacity?: number
}

type NiivueVolumeViewerProps = {
  volumeUrls?: string[]
  volumes?: VolumeConfig[]
  syncPeers?: NiivueViewerLike[]
  onViewerReady?: (viewer: NiivueViewerLike) => void
  sliceIndex?: number
  sliceCount?: number
  className?: string
  style?: React.CSSProperties
}

const NiivueVolumeViewer: React.FC<NiivueVolumeViewerProps> = ({
  volumeUrls = [],
  volumes,
  syncPeers,
  onViewerReady,
  sliceIndex,
  sliceCount,
  className,
  style,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const resolvedVolumes = volumes ?? volumeUrls.map((url) => ({ url }))

  useNiivueViewer({
    canvasRef,
    volumes: resolvedVolumes,
    syncPeers,
    onViewerReady,
    sliceIndex,
    sliceCount,
  })

  return <canvas ref={canvasRef} className={className} style={{ width: '100%', height: '100%', ...style }} />
}

export default NiivueVolumeViewer
export type { NiivueVolumeViewerProps }
