import numpy as np
import ot


def empirical_wasserstein_distance(
    samples_hat: np.ndarray,
    samples_target: np.ndarray,
    max_points: int = 2048,
) -> float:
    samples_hat = np.asarray(samples_hat, dtype=np.float64)
    samples_target = np.asarray(samples_target, dtype=np.float64)

    n_hat = samples_hat.shape[0]
    n_target = samples_target.shape[0]

    if n_hat > max_points:
        idx = np.random.choice(n_hat, max_points, replace=False)
        samples_hat = samples_hat[idx]
        n_hat = max_points
    if n_target > max_points:
        idx = np.random.choice(n_target, max_points, replace=False)
        samples_target = samples_target[idx]
        n_target = max_points

    a = np.full(n_hat, 1.0 / n_hat, dtype=np.float64)
    b = np.full(n_target, 1.0 / n_target, dtype=np.float64)
    cost = ot.dist(samples_hat, samples_target, metric="euclidean")
    cost = cost * cost
    wasserstein_squared = ot.emd2(a, b, cost)
    return float(np.sqrt(max(wasserstein_squared, 0.0)))