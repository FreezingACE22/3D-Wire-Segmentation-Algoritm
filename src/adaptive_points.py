import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s else default


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def turning_angles_deg(t: np.ndarray) -> np.ndarray:
    dots = np.sum(t[:-1] * t[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def make_bend_mask(angles_deg: np.ndarray, angle_thresh_deg: float, spread: int = 3) -> np.ndarray:
    n = angles_deg.shape[0] + 1
    bend = np.zeros(n, dtype=bool)

    idx = np.where(angles_deg >= angle_thresh_deg)[0]
    for i in idx:
        bend[i] = True
        bend[i + 1] = True

    if spread > 0:
        bend2 = bend.copy()
        for i in range(n):
            if bend[i]:
                lo = max(0, i - spread)
                hi = min(n, i + spread + 1)
                bend2[lo:hi] = True
        bend = bend2

    return bend


def adaptive_resample(points: np.ndarray,
                      bend_mask: np.ndarray,
                      straight_spacing: float,
                      bend_spacing: float) -> np.ndarray:
    n = points.shape[0]
    if n <= 1:
        return np.arange(n, dtype=int)

    keep = [0]
    acc_dist = 0.0

    for i in range(1, n - 1):
        acc_dist += float(np.linalg.norm(points[i] - points[i - 1]))
        spacing = bend_spacing if bend_mask[i] else straight_spacing
        if acc_dist >= spacing:
            keep.append(i)
            acc_dist = 0.0

    keep.append(n - 1)
    return np.array(keep, dtype=int)


def main():
    print("=== Curvature-based resampling + 3D plot ===")
    in_csv = ask("Input CSV path", r"data\csv\centerline.csv")
    out_csv = ask("Output CSV path", r"data\csv\centerline_adaptive.csv")

    angle_thresh_deg = float(ask("Angle threshold (deg) for bend detection", "0.5"))
    straight_spacing = float(ask("Spacing on straight segments (mm)", "10"))
    bend_spacing = float(ask("Spacing near bends/fillets (mm)", "1"))
    spread = int(ask("Bend spread (extra points around bends)", "4"))

    show_plot = ask("Show plot? (y/n)", "y").lower().startswith("y")
    save_plot = ask("Save plot image? (empty = no)", "")

    # Ensure output folder exists
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    for c in ["x", "y", "z", "dx", "dy", "dz"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_csv}")

    pts = df[["x", "y", "z"]].to_numpy(dtype=float)
    t = unit(df[["dx", "dy", "dz"]].to_numpy(dtype=float))

    ang = turning_angles_deg(t)
    bend_mask = make_bend_mask(ang, angle_thresh_deg, spread=spread)

    keep_idx = adaptive_resample(pts, bend_mask, straight_spacing, bend_spacing)

    df_out = df.iloc[keep_idx].copy()
    df_out["bend"] = bend_mask[keep_idx].astype(int)
    df_out.to_csv(out_csv, index=False)

    print(f"\nInput points : {len(df)}")
    print(f"Output points: {len(df_out)}")
    print(f"Saved        : {out_csv}")

    # Plot raw vs resampled
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=1, label="raw")

    rp = df_out[["x", "y", "z"]].to_numpy(dtype=float)
    ax.scatter(rp[:, 0], rp[:, 1], rp[:, 2], s=10, label="resampled")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if save_plot:
        Path(save_plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot, dpi=200)
        print(f"Plot saved   : {save_plot}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
