export interface NiivueSyncPeer {
  broadcastTo?: (target: NiivueSyncPeer) => void
}

export const synchronizeNiivuePeers = (source: NiivueSyncPeer, peers: NiivueSyncPeer[]): void => {
  if (typeof source.broadcastTo !== 'function') {
    return
  }

  peers.forEach((peer) => {
    if (peer && peer !== source) {
      source.broadcastTo?.(peer)
    }
  })
}
