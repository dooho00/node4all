from __future__ import annotations

import os
from typing import Optional, Sequence, List

import numpy as np

def save_loss_curve(
    *,
    out_path: str,
    epochs: Sequence[int],
    train_loss: Sequence[float],
    train_link_loss: Optional[Sequence[float]] = None,
    train_repr_penalty: Optional[Sequence[float]] = None,
    val_metric: Optional[Sequence[Optional[float]]] = None,
    metric_name: str = "val",
    title: str = "",
    early_stopped: bool = False,
    early_stop_epoch: Optional[int] = None,
    best_epoch: Optional[int] = None,
) -> bool:
    """
    Save a training loss curve plot (and optional metric) to out_path.

    Returns True on success, False if plotting is unavailable or inputs are empty.
    """
    if not out_path:
        return False
    if not epochs or not train_loss:
        return False

    n = min(len(epochs), len(train_loss))
    if n <= 0:
        return False

    epochs = list(epochs)[:n]
    train_loss = list(train_loss)[:n]
    train_link_loss = list(train_link_loss)[:n] if train_link_loss is not None else None
    train_repr_penalty = list(train_repr_penalty)[:n] if train_repr_penalty is not None else None
    val_metric = list(val_metric)[:n] if val_metric is not None else None

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    ax.plot(epochs, train_loss, label="train/loss", linewidth=1.8)
    if train_link_loss is not None:
        ax.plot(epochs, train_link_loss, label="train/link", linewidth=1.2, alpha=0.9)
    if train_repr_penalty is not None:
        ax.plot(epochs, train_repr_penalty, label="train/repr_pen", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)

    ax2 = None
    if val_metric is not None:
        xs: List[int] = []
        ys: List[float] = []
        for e, v in zip(epochs, val_metric):
            if v is None:
                continue
            xs.append(int(e))
            ys.append(float(v))
        if xs:
            ax2 = ax.twinx()
            ax2.plot(xs, ys, label=f"val/{metric_name}", color="tab:green", linewidth=1.6, alpha=0.9)
            ax2.set_ylabel(metric_name)

    if best_epoch is not None:
        ax.axvline(int(best_epoch), color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            int(best_epoch),
            0.98,
            f"best@{int(best_epoch)}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
            color="tab:blue",
        )

    if early_stopped and early_stop_epoch is not None:
        ax.axvline(int(early_stop_epoch), color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            int(early_stop_epoch),
            0.02,
            f"early-stop@{int(early_stop_epoch)}",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color="tab:red",
        )

    status = "early-stopped" if early_stopped else "completed"
    full_title = f"{title} ({status})" if title else f"loss_curve ({status})"
    ax.set_title(full_title)

    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)
    if handles:
        ax.legend(handles, labels, loc="best", fontsize=9, frameon=True)
    
    loss_arr = np.asarray(train_loss, dtype=float)

    # Robust upper bound suppresses early spikes
    p_low, p_high = 0.0, 100.0
    robust_low, robust_high = np.percentile(loss_arr, [p_low, p_high])

    # Ensure the true minimum loss is always visible
    y_min = min(loss_arr.min(), robust_low)
    y_max = robust_high

    # Padding for readability
    pad = 0.1 * (y_max - y_min + 1e-12)
    ax.set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    try:
        fig.savefig(out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return True
