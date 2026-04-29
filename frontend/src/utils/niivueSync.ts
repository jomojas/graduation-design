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

  // 中文说明：Niivue 的 broadcastTo 期望一次传入“目标 viewer 数组”。
  // 如果对每个 peer 单独调用 broadcastTo，Niivue 内部可能会覆盖 sync 列表，
  // 导致只有最后一个 peer 能收到同步更新。
  source.broadcastTo(targets)
}
