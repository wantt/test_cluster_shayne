# -*- coding: utf-8 -*-
import numpy as np

def compute_sse(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    k = centroids.shape[0]
    sse = 0.0
    for c in range(k):
        mask = (labels == c)
        if not np.any(mask):
            continue
        diffs = points[mask] - centroids[c]
        sse += float(np.sum(diffs * diffs))
    return sse
