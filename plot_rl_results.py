import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple


def _moving_stats(data: Sequence[float], window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute moving average / std over 1D data. Returns (indices, avg, std).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n == 0:
        raise ValueError("data must be non-empty")

    if window >= n:
        return np.array([(n - 1) / 2.0]), np.array([x.mean()]), np.array([x.std()])

    w = np.ones(window) / window
    mov_avg = np.convolve(x, w, mode="valid")
    mov_std = np.asarray([x[i : i + window].std() for i in range(n - window + 1)])

    half = window // 2
    if window % 2 == 0:
        idx = np.arange(half - 0.5, n - half + 0.5)
    else:
        idx = np.arange(half, n - half)
    return idx, mov_avg, mov_std


def plot_training_statistics(
    episode_rewards: Sequence[float],
    episode_lengths: Sequence[int],
    *,
    window: int = 50,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 200,
    figsize: Tuple[float, float] = (10, 8),
    show: bool = True,
) -> None:
    """
    Plot training episode rewards and episode lengths stacked vertically.
    """
    if len(episode_rewards) != len(episode_lengths):
        raise ValueError("episode_rewards and episode_lengths must have the same length")

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if title:
        fig.suptitle(title)

    # --- Rewards subplot ---
    ax = axes[0]
    episodes = np.arange(len(episode_rewards))
    ax.plot(episodes, episode_rewards, alpha=0.35, linewidth=1.0, label="Raw")
    idx_r, avg_r, std_r = _moving_stats(episode_rewards, window)
    ax.plot(idx_r, avg_r, linewidth=2.0, label=f"{window}-ep MA")
    ax.fill_between(idx_r, avg_r - std_r, avg_r + std_r, alpha=0.2, label="±1 std")
    ax.set_title("Episode Reward")
    ax.set_ylabel("Total Reward")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    # --- Lengths subplot ---
    ax = axes[1]
    episodes_len = np.arange(len(episode_lengths))
    ax.plot(episodes_len, episode_lengths, alpha=0.35, linewidth=1.0, label="Raw")
    idx_l, avg_l, std_l = _moving_stats(episode_lengths, window)
    ax.plot(idx_l, avg_l, linewidth=2.0, label=f"{window}-ep MA")
    ax.fill_between(idx_l, avg_l - std_l, avg_l + std_l, alpha=0.2, label="±1 std")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    fig.tight_layout()
    if title:
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return
