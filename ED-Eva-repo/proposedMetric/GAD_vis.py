import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from pathlib import Path
import torch


def load_pickle_auto_device(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b),
                    map_location=device,
                    weights_only=False
                )
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return DeviceUnpickler(f).load()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_shape(pred):
    # Accept (N, K, T, D) or (K, T, D); return (N, K, T, D)
    if pred.ndim == 3:
        pred = pred[None, ...]
    return pred


def gad_score(xy, max_k=4):
    """
    xy: shape (N, K, T, 2)
        N = prediction index (e.g., agents)
        K = ensemble samples for each prediction
        T = time steps
    Returns scalar GAD.
    """
    N, K, T, D = xy.shape
    total = 0.0

    for n in range(N):
        for t in range(T):
            data = xy[n, :, t, :]  # (K, 2)

            # If there are too few samples, cap mixture count
            cur_max_k = min(max_k, len(data))
            bics, gmms = [], []

            for k in range(1, cur_max_k + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=0
                )
                gmm.fit(data)
                bics.append(gmm.bic(data))
                gmms.append(gmm)

            best = gmms[np.argmin(bics)]
            pi, mus, covs = best.weights_, best.means_, best.covariances_

            # Mixture mean
            mu_mix = (pi[:, None] * mus).sum(axis=0)

            # Mixture covariance
            Sigma = np.zeros((D, D))
            for i in range(len(pi)):
                Sigma += pi[i] * covs[i]
                dmu = (mus[i] - mu_mix)[:, None]
                Sigma += pi[i] * (dmu @ dmu.T)

            det_sigma = np.linalg.det(Sigma).real
            det_sigma = max(det_sigma, 0.0)  # numerical safety
            total += np.sqrt(det_sigma)

    return total / (N * T)


def plot_kde_with_gad(ax, xy, title, max_k=4):
    """
    xy: (N, K, T, 2)
    KDE visualization uses all points pooled together.
    """
    pts = xy.reshape(-1, 2).T

    kde = gaussian_kde(pts)
    xmin, ymin = pts.min(axis=1) - 0.1
    xmax, ymax = pts.max(axis=1) + 0.1

    # Use 100j for speed; change back to 200j if needed
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    flat = Z.ravel()
    levels = np.percentile(flat, [10, 25, 50, 75, 90])

    cf = ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.4)
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=1)

    # Optional: overlay all trajectories
    flat_xy = xy.reshape(-1, xy.shape[2], 2)
    for traj in flat_xy:
        ax.plot(traj[:, 0], traj[:, 1], linestyle=":", color="gray", alpha=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    gad = gad_score(xy, max_k=max_k)
    ax.set_title(f"{title}\nGAD = {gad:.4f}")
    return cf, gad


configs = [
    {
        "input": "./exampleData/scenario_1_output_imputed.pkl",
        "title": "Concentrated Predictions",
        "save": "./proposedMetric/myUse/update/paper_concentrated_gad.png"
    },
    {
        "input": "./exampleData/scenario_1_output.pkl",
        "title": "Diverse Predictions",
        "save": "./proposedMetric/myUse/update/paper_diverse_gad.png"
    }
]

for cfg in configs:
    output = load_pickle_auto_device(cfg["input"])

    pred = to_numpy(output["predicted_trajectory"])

    # Use first scenario, then ensure shape (N, K, T, D)
    pred0 = pred[0]
    pred0 = _ensure_shape(pred0)
    xy = pred0[..., :2]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf, gad = plot_kde_with_gad(ax, xy, cfg["title"], max_k=4)
    fig.colorbar(cf, ax=ax, label="Density")

    plt.tight_layout()
    Path(cfg["save"]).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["save"], dpi=300)
    plt.show()
    plt.close(fig)