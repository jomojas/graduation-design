export interface NiivueSyncPeer {
  broadcastTo?: (targets: NiivueSyncPeer[], options?: Record<string, unknown>) => void
}

export const synchronizeNiivuePeers = (source: NiivueSyncPeer, peers: NiivueSyncPeer[]): void => {
  if (typeof source.broadcastTo !== 'function') {
    return
  }

  const targets = peers.filter((peer): peer is NiivueSyncPeer => Boolean(peer) && peer !== source)
  if (targets.length === 0) {
    return
  }

  // Niivue's broadcastTo expects an array of peers; calling it repeatedly can
  // overwrite the internal sync list (only the last peer receives updates).
  source.broadcastTo(targets)
}
