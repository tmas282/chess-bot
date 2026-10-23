import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def moving_average(data, window_size=20):
    """Calculates rolling mean to smooth noisy RL metrics."""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_training_metrics(rewards, a2c_losses, window_size=20, save_path="training_curves.png"):
    if len(a2c_losses) == 0 and len(rewards) == 0:
        print("No metrics recorded to plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Episode Rewards
    if len(rewards) > 0:
        episodes_r = range(1, len(rewards) + 1)
        axes[0].plot(episodes_r, rewards, alpha=0.3, color="green", label="Raw Reward")
        if len(rewards) >= window_size:
            smooth_rewards = moving_average(rewards, window_size)
            axes[0].plot(range(window_size, len(rewards) + 1), smooth_rewards, color="green", linewidth=2, label=f"MA ({window_size})")
        axes[0].set_ylabel("Reward")
        axes[0].set_xlabel("Completed Games")
        axes[0].set_title("Training Metrics")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.6)
    else:
        axes[0].text(0.5, 0.5, "No games completed yet", ha="center", va="center")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Metrics")

    # 2. A2C Loss
    if len(a2c_losses) > 0:
        updates_l = range(1, len(a2c_losses) + 1)
        axes[1].plot(updates_l, a2c_losses, alpha=0.3, color="red", label="Raw A2C Loss")
        if len(a2c_losses) >= window_size:
            smooth_a2c = moving_average(a2c_losses, window_size)
            axes[1].plot(range(window_size, len(a2c_losses) + 1), smooth_a2c, color="red", linewidth=2, label=f"MA ({window_size})")
        axes[1].set_ylabel("A2C Loss")
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
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    loss: float | None = None,
    save_dir: str = "checkpoints",
    filename: str | None = None,
) -> str:
    """
    Saves a complete training checkpoint following the PyTorch recommended way.
    Saves actor state_dict, critic state_dict, optimizer state_dict, and training metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_ep_{episode}.pt"

    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        "episode": episode,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"--> Checkpoint saved to: {os.path.abspath(filepath)}")
    return filepath


def load_checkpoint(
    filepath: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> int:
    """
    Loads a checkpoint into actor, critic, and optionally optimizer.
    Returns the episode number from the checkpoint.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_episode = checkpoint.get("episode", 0)
    print(f"--> Checkpoint loaded from: {filepath} (Episode {start_episode})")
    return start_episode
