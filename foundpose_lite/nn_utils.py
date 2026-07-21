import numpy as np


def batched_soft_assign(descs: np.ndarray, words: np.ndarray, soft_k: int, sigma: float,
                         batch_size: int = 2000):
    """descs: (P, D), words: (W, D). Returns nearest (P, soft_k) int64, weights (P, soft_k) float32.
    Uses the ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b expansion, batched over P, to avoid
    materializing a (P, W, D) tensor which blows up memory for large P/W/D."""
    p = descs.shape[0]
    words_sq = np.sum(words ** 2, axis=1)  # (W,)
    nearest_all = np.zeros((p, soft_k), dtype=np.int64)
    weights_all = np.zeros((p, soft_k), dtype=np.float32)
    for start in range(0, p, batch_size):
        end = min(start + batch_size, p)
        chunk = descs[start:end]  # (B, D)
        chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)  # (B, 1)
        dists_sq = chunk_sq + words_sq[None, :] - 2 * chunk @ words.T  # (B, W)
        np.clip(dists_sq, 0, None, out=dists_sq)
        nearest = np.argsort(dists_sq, axis=1)[:, :soft_k]
        nearest_dists_sq = np.take_along_axis(dists_sq, nearest, axis=1)
        weights = np.exp(-nearest_dists_sq / (2 * sigma ** 2))
        nearest_all[start:end] = nearest
        weights_all[start:end] = weights
    return nearest_all, weights_all
