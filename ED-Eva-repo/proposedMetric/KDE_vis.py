import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
                    # weights_only=False
                )
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return DeviceUnpickler(f).load()


# Configuration for the two datasets
configs = [
    {
        "input": "./exampleData/scenario_1_output_imputed.pkl",
        "title": "Concentrated Predictions",
        "save": "./proposedMetric/myUse/update/code_concentrated_kde.png"
    },
    {
        "input": "./exampleData/scenario_1_output.pkl",
        "title": "Diverse Predictions",
        "save": "./proposedMetric/myUse/update/code_diverse_kde.png"
    }
]

for cfg in configs:
    # 1) Load pickle with automatic device mapping
    output = load_pickle_auto_device(cfg["input"])

    # 2) Extract and convert to numpy
    pred = output["predicted_trajectory"]
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)

    # 3) Pick first scenario and XY coords
    pred0 = pred[0]         # (num_trajs, T, D)
    xy = pred0[..., :2]     # (num_trajs, T, 2)

    # 4) Build KDE over all predicted points
    points = xy.reshape(-1, 2).T  # (2, num_trajs*T)
    kde = gaussian_kde(points)

    # 5) Create grid and evaluate KDE
    xmin, ymin = points.min(axis=1) - 0.1
    xmax, ymax = points.max(axis=1) + 0.1
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # 6) (Optional) Simulate obs & gt for demo
    T = xy.shape[1]
    obs = np.stack([np.linspace(0, 1, T), np.zeros(T)], axis=1)
    gt = obs + np.sin(np.linspace(0, np.pi / 2, T))[:, None] * 0.2

    # 7) Compute the specific percentile levels
    flat = Z.ravel()
    levels = np.percentile(flat, [10, 25, 50, 75, 90])

    # 8) Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # 8.1) Filled shading
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.4)

    # 8.2) Contour lines at percentiles
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=1)
    fig.colorbar(cf, ax=ax, label="Density")

    # 8.3) Overlay predicted trajectories
    for traj in xy:
        ax.plot(traj[:, 0], traj[:, 1], linestyle=":", color="gray")

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{cfg['title']}\nTrajectory Prediction with KDE Uncertainty")

    # save and show
    plt.tight_layout()
    save_path = Path(cfg["save"]).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"Please check {save_path} to view the generated image.")
