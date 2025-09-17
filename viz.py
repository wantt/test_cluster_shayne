# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from polar import  plot_coordinate_comparison   
def save_scatter_snapshot(points: np.ndarray,
                          labels: np.ndarray,
                          centroids: np.ndarray = None,
                          out_path: str = "snapshot.png",
                          title: str = None):
    assert points.shape[1] == 2, "Points must be 2D for this visualization."
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, s=5, alpha=0.7)
    if centroids is not None:
        centroids = np.asarray(centroids)
        if centroids.size > 0:
            plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="x")
    if title:
        plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


    plot_coordinate_comparison(points, polar_points=None, labels=labels, centroids=centroids,
                          out_path=out_path.replace(".png", "_polar.png"),
                          title=title)
