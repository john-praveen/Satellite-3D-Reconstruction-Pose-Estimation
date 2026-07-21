import numpy as np


def cyclic_buddies_matching(query_points_2d: np.ndarray, query_descs: np.ndarray,
                             template_descs: np.ndarray, top_k: int):
    """Ports FoundPose's "cyclic buddies" correspondence filter (arxiv.org/abs/2204.03635):
    for each query patch, find its nearest template patch, then from that template patch
    find ITS nearest query patch (round trip). If the round trip lands back near the
    original 2D query location, the correspondence is self-consistent and trustworthy.
    Keep only the `top_k` most self-consistent correspondences, discarding the rest --
    this feeds PnP-RANSAC a much cleaner, pre-filtered candidate set instead of every
    foreground patch (which is what a plain nearest-neighbor match does). `top_k` must be
    meaningfully smaller than the typical foreground patch count to actually act as a
    filter (e.g. 60 for crops with ~130-150 valid patches) -- if it's larger than the
    available patch count, this is a no-op.

    query_points_2d: (Q, 2), query_descs: (Q, D), template_descs: (T, D).
    Returns: query_ids (K,), template_ids (K,) -- indices into the input arrays.
    """
    # (Q, T) pairwise distances -- fine at this scale (a few hundred x a few hundred).
    dists = np.linalg.norm(query_descs[:, None, :] - template_descs[None, :, :], axis=2)

    query2tmpl_nn = np.argmin(dists, axis=1)       # (Q,) nearest template patch per query patch
    tmpl2query_nn = np.argmin(dists, axis=0)       # (T,) nearest query patch per template patch

    cycle_ids = tmpl2query_nn[query2tmpl_nn]       # (Q,) round-trip query index
    cycle_dists = np.linalg.norm(query_points_2d - query_points_2d[cycle_ids], axis=1)

    top_k = min(top_k, len(query_points_2d))
    query_ids = np.argsort(cycle_dists)[:top_k]
    template_ids = query2tmpl_nn[query_ids]
    return query_ids, template_ids
