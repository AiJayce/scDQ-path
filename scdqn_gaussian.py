"""Density-stratified latent sampling and expression cosine comparisons."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


DENSITY_BANDS = ((0.8, 1.0), (0.6, 0.8), (0.4, 0.6))
BAND_PROPORTIONS = np.array([0.683, 0.271, 0.046])


def band_counts(total):
    if not isinstance(total, (int, np.integer)) or total < 1:
        raise ValueError("Sample total must be a positive integer.")
    expected = total * BAND_PROPORTIONS
    counts = np.floor(expected).astype(int)
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:total - counts.sum()]] += 1
    return counts


def sample_density_bands(centers, observed_sd, pca_mean, components,
                         density, x_edges, y_edges, total=100, sigma_sd=0.03,
                         seed=20261031, max_candidates=1_000_000, batch_size=2048):
    centers = np.asarray(centers, dtype=np.float64)
    observed_sd = np.asarray(observed_sd, dtype=np.float64)
    pca_mean = np.asarray(pca_mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    x_edges, y_edges = np.asarray(x_edges), np.asarray(y_edges)
    if centers.ndim != 2 or len(centers) == 0:
        raise ValueError("Gaussian centers must contain post-start latent visits.")
    dimension = centers.shape[1]
    if observed_sd.shape != (dimension,) or pca_mean.shape != (dimension,) or components.shape != (2, dimension):
        raise ValueError("Latent SD or original PCA dimensions do not match visits.")
    if not all(np.isfinite(v).all() for v in (centers, observed_sd, pca_mean, components, density, x_edges, y_edges)):
        raise ValueError("Non-finite sampling inputs.")
    if np.any(observed_sd <= 0) or sigma_sd <= 0 or not np.isfinite(sigma_sd):
        raise ValueError("Gaussian latent SD must be positive and finite.")
    if density.shape != (len(y_edges) - 1, len(x_edges) - 1):
        raise ValueError("Density grid shape does not match PCA edges.")
    if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0):
        raise ValueError("Density edges must be strictly increasing.")
    if np.min(density) < 0 or not np.isclose(density.max(), 1) or density.max() > 1 + 1e-12:
        raise ValueError("Density must be normalized to its maximum, not quantiles.")
    if max_candidates < 1 or batch_size < 1:
        raise ValueError("Candidate limits must be positive.")
    interpolate = RegularGridInterpolator(
        ((y_edges[1:] + y_edges[:-1]) / 2, (x_edges[1:] + x_edges[:-1]) / 2),
        density, bounds_error=False, fill_value=0,
    )
    targets = band_counts(total)
    latent_parts, xy_parts, density_parts = [[] for _ in targets], [[] for _ in targets], [[] for _ in targets]
    accepted = np.zeros(3, dtype=int)
    rng = np.random.default_rng(seed)
    proposed = 0
    while np.any(accepted < targets) and proposed < max_candidates:
        size = min(batch_size, max_candidates - proposed)
        z = centers[rng.integers(len(centers), size=size)] + rng.normal(size=(size, dimension)) * (sigma_sd * observed_sd)
        z = z.astype(np.float32)
        xy = (z - pca_mean) @ components.T
        relative = interpolate(xy[:, ::-1])
        proposed += size
        for band, (lower, upper) in enumerate(DENSITY_BANDS):
            valid = (relative >= lower) & ((relative <= upper) if band == 0 else (relative < upper))
            selected = np.flatnonzero(valid)[:targets[band] - accepted[band]]
            if len(selected):
                latent_parts[band].append(z[selected])
                xy_parts[band].append(xy[selected])
                density_parts[band].append(relative[selected])
                accepted[band] += len(selected)
    if not np.array_equal(accepted, targets):
        raise RuntimeError(f"Density bands not filled after {proposed} candidates: {accepted.tolist()} / {targets.tolist()}.")
    z = np.concatenate([np.concatenate(parts) for parts, count in zip(latent_parts, targets) if count])
    xy = np.concatenate([np.concatenate(parts) for parts, count in zip(xy_parts, targets) if count])
    relative = np.concatenate([np.concatenate(parts) for parts, count in zip(density_parts, targets) if count])
    labels = np.repeat(np.arange(3), targets)
    return z, xy, relative, labels, targets, proposed


def expression_cosines(expression, labels):
    expression = np.asarray(expression, dtype=np.float64)
    labels = np.asarray(labels)
    if expression.ndim != 2 or labels.shape != (len(expression),):
        raise ValueError("Expected sample-by-gene expression and matching band labels.")
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("Expression must be finite and nonnegative.")
    if not np.array_equal(np.unique(labels), np.arange(3)):
        raise ValueError("All three density bands must be present.")
    means = np.stack([expression[labels == band].mean(axis=0) for band in range(3)])

    def cosine_rows(values):
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("Cosine similarity is undefined for zero or non-finite expression norms.")
        unit = values / norms[:, None]
        return np.clip(unit @ unit.T, 0, 1)

    return means, cosine_rows(means), cosine_rows(expression)
