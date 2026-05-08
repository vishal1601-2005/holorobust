import numpy as np
from scipy.stats import gaussian_kde


def tv_distance_test(X_normal, n_subpops=5,
                     n_samples=1000, threshold=0.05):
    """
    Total Variation distance test for HoloRobust suitability.

    Measures distributional consistency of the normal/background
    data. Low TV distance indicates a structured background
    distribution suitable for holographic regularization.

    Parameters
    ----------
    X_normal   : np.ndarray (n, d) normal/background samples
    n_subpops  : int, number of subpopulations to compare
    n_samples  : int, samples per subpopulation
    threshold  : float, TV distance threshold
                 below = structured (HoloRobust recommended)
                 above = heterogeneous (SpectralNorm AE recommended)

    Returns
    -------
    dict with keys: mean_tv, std_tv, suitable, recommendation

    Validated thresholds from paper benchmarks:
        LHCO HEP jets      TV=0.037  HoloRobust  +1.40% AUC
        SMAP telemetry     TV=0.031  HoloRobust  +0.34% AUC
        CIC-IDS2017 traffic TV=0.061 SpectralNorm -4.45% AUC
    """
    n, d    = X_normal.shape
    indices = np.random.permutation(n)
    subpops = np.array_split(
        indices[:n_subpops * n_samples], n_subpops)

    tv_distances = []
    for i in range(n_subpops):
        for j in range(i + 1, n_subpops):
            X_i = X_normal[subpops[i]]
            X_j = X_normal[subpops[j]]
            feature_tvs = []
            feat_sample = np.random.choice(
                d, min(d, 10), replace=False)
            for f in feat_sample:
                xi   = X_i[:, f]
                xj   = X_j[:, f]
                grid = np.linspace(
                    min(xi.min(), xj.min()),
                    max(xi.max(), xj.max()), 200)
                try:
                    kde_i = gaussian_kde(xi)(grid)
                    kde_j = gaussian_kde(xj)(grid)
                    kde_i /= kde_i.sum()
                    kde_j /= kde_j.sum()
                    tv    = 0.5 * np.abs(kde_i - kde_j).sum()
                    feature_tvs.append(tv)
                except Exception:
                    pass
            if feature_tvs:
                tv_distances.append(np.mean(feature_tvs))

    mean_tv  = float(np.mean(tv_distances))
    std_tv   = float(np.std(tv_distances))
    suitable = mean_tv < threshold

    return {
        "mean_tv":        round(mean_tv, 4),
        "std_tv":         round(std_tv, 4),
        "suitable":       suitable,
        "threshold":      threshold,
        "recommendation": (
            "HoloRobust RECOMMENDED (structured background)"
            if suitable else
            "SpectralNorm AE RECOMMENDED (heterogeneous background)"
        ),
    }
