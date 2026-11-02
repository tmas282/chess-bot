import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class DeviceManager:
    """
    Singleton for managing and caching the PyTorch device (CUDA or CPU).
    """
    _device: torch.device | None = None

    @classmethod
    def get_device(cls) -> torch.device:
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._device

    @classmethod
    def set_device(cls, device: torch.device | str) -> None:
        cls._device = torch.device(device)


get_device = DeviceManager.get_device


def moving_average(data, window_size=20):
    """Calculates rolling mean to smooth noisy RL metrics."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_training_metrics(rewards, value_losses, window_size=20, save_path="training_curves.png"):
    if len(value_losses) == 0 and len(rewards) == 0:
        print("No metrics recorded to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Episode Game Outcomes / Rewards
    if len(rewards) > 0:
        episodes_r = range(1, len(rewards) + 1)
        axes[0].plot(episodes_r, rewards, alpha=0.3, color="green", label="Game Reward (White Outcome)")
        if len(rewards) >= window_size:
            smooth_rewards = moving_average(rewards, window_size)
            axes[0].plot(range(window_size, len(rewards) + 1), smooth_rewards, color="green", linewidth=2, label=f"MA ({window_size})")
        axes[0].set_ylabel("Outcome Reward")
        axes[0].set_xlabel("Completed Games")
        axes[0].set_title("Training Metrics")
        axes[0].axhline(0, color="gray", linestyle=":", alpha=0.7)
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, "No games completed yet", ha="center", va="center")
        axes[0].set_ylabel("Outcome Reward")
        axes[0].set_title("Training Metrics")

    # 2. Value Iteration Loss (MSE)
    if len(value_losses) > 0:
        updates_l = range(1, len(value_losses) + 1)
        axes[1].plot(updates_l, value_losses, alpha=0.3, color="blue", label="Raw Value Loss (MSE)")
        if len(value_losses) >= window_size:
            smooth_loss = moving_average(value_losses, window_size)
            axes[1].plot(range(window_size, len(value_losses) + 1), smooth_loss, color="blue", linewidth=2, label=f"MA ({window_size})")
        axes[1].set_ylabel("Value Loss (MSE)")
        axes[1].set_xlabel("Update Step")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"--> Plot saved successfully to: {os.path.abspath(save_path)}")
    plt.close(fig)


def save_checkpoint(
    episode: int,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    loss: float | None = None,
    save_dir: str = "checkpoints",
    filename: str | None = None,
) -> str:
    """
    Saves a training checkpoint for ValueViT.
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_ep_{episode}.pt"

    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        "episode": episode,
        "value_net_state_dict": value_net.state_dict(),
        "loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"--> Checkpoint saved to: {os.path.abspath(filepath)}")
    return filepath


def load_checkpoint(
    filepath: str,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> int:
    """
    Loads a checkpoint into value_net, and optionally optimizer.
    Returns the episode number from the checkpoint.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    value_net.load_state_dict(checkpoint["value_net_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_episode = checkpoint.get("episode", 0)
    print(f"--> Checkpoint loaded from: {filepath} (Episode {start_episode})")
    return start_episode
