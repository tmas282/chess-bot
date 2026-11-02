import glob
import os
import time

import chess
import kagglehub
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from agent.environment import ChessEnvironment
from agent.value_vit import ValueViT
from utils import get_device, save_checkpoint

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 5
VAL_SPLIT = 0.1
DATASET_DIR = "dataset"
CHECKPOINT_DIR = "checkpoints"
DTYPE = torch.bfloat16
DEVICE = get_device()


# =============================================================================
# Data Download & Preprocessing
# =============================================================================
def prepare_dataset(dataset_dir: str = DATASET_DIR) -> str:
    """
    Downloads dataset files directly to the destination folder using kagglehub's
    output_dir argument without needing any manual copy or move operations.
    """
    parquet_pattern = os.path.join(dataset_dir, "*.parquet")
    if glob.glob(parquet_pattern):
        print(f"Dataset already present in ./{dataset_dir}")
        return dataset_dir

    print(f"Downloading dataset directly into ./{dataset_dir} via kagglehub...")
    path = kagglehub.dataset_download(
        "mateuszgrzybpl/lichess-chess-positions-ml-ready-and-deduplicated",
        output_dir=dataset_dir,
    )
    print("Dataset ready at:", path)
    return path


def load_mate_in_one_data(dataset_dir: str = DATASET_DIR) -> tuple[list[str], list[float]]:
    """
    Scans parquet files using Polars and filters for mate-in-1 positions.
    In this dataset, evaluations are already White-centric:
      +1: White has forced mate (+1.0)
      -1: Black has forced mate (-1.0)
    """
    parquet_pattern = os.path.join(dataset_dir, "*.parquet")
    print(f"Reading and filtering dataset from '{parquet_pattern}' with Polars...")

    df = (
        pl.scan_parquet(parquet_pattern)
        .filter(pl.col("mate").is_in([1, -1]))
        .select([
            pl.col("fen"),
            pl.col("mate").cast(pl.Float32).alias("target"),
        ])
        .collect()
    )

    total_positions = len(df)
    white_wins = (df["target"] == 1.0).sum()
    black_wins = (df["target"] == -1.0).sum()

    print(f"Extracted {total_positions:,} mate-in-1 positions:")
    print(f"  White wins (+1.0): {white_wins:,}")
    print(f"  Black wins (-1.0): {black_wins:,}")

    fens = df["fen"].to_list()
    targets = df["target"].to_list()
    return fens, targets


class ChessMateDataset(Dataset):
    """
    Dataset converting FEN strings into early-fusion (17, 8, 8) board state tensors.
    """

    def __init__(self, fens: list[str], targets: list[float], dtype: torch.dtype = DTYPE):
        self.fens = fens
        self.targets = targets
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        board = chess.Board(self.fens[idx])
        state_tensor = ChessEnvironment.board_to_tensor(board, device="cpu", dtype=self.dtype)
        target_tensor = torch.tensor([self.targets[idx]], dtype=self.dtype)
        return state_tensor, target_tensor


# =============================================================================
# Training
# =============================================================================
def train():
    print(f"=== Starting Supervised Pretraining on {DEVICE} ({DTYPE}) ===")
    prepare_dataset(DATASET_DIR)
    fens, targets = load_mate_in_one_data(DATASET_DIR)

    full_dataset = ChessMateDataset(fens, targets, dtype=DTYPE)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    avg_val_loss = 0.0
    print(f"Dataset split: {train_size:,} train samples | {val_size:,} validation samples")

    model = ValueViT().to(DEVICE).to(DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_total = 0.0

        for step, (states, targets_batch) in enumerate(train_loader, start=1):
            states = states.to(DEVICE, non_blocking=True)
            targets_batch = targets_batch.to(DEVICE, non_blocking=True)

            model.reset_noise()
            preds = model(states)
            loss = loss_fn(preds, targets_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                current_avg = train_loss_total / step
                print(
                    f"Epoch [{epoch}/{EPOCHS}] | Step [{step}/{len(train_loader)}] | "
                    f"Train Loss (MSE): {current_avg:.4f}"
                )

        avg_train_loss = train_loss_total / len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for states, targets_batch in val_loader:
                states = states.to(DEVICE, non_blocking=True)
                targets_batch = targets_batch.to(DEVICE, non_blocking=True)
                preds = model(states)
                val_loss_total += loss_fn(preds, targets_batch).item()

        avg_val_loss = val_loss_total / len(val_loader)
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"--- Epoch {epoch} Completed in {epoch_time:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---"
        )

        # Save epoch checkpoint
        save_checkpoint(
            episode=epoch,
            value_net=model,
            optimizer=optimizer,
            loss=avg_val_loss,
            save_dir=CHECKPOINT_DIR,
            filename=f"checkpoint_supervised_epoch_{epoch}.pt",
        )

    # Save final pretrained weights
    save_checkpoint(
        episode=EPOCHS,
        value_net=model,
        optimizer=optimizer,
        loss=avg_val_loss,
        save_dir=CHECKPOINT_DIR,
        filename="checkpoint_supervised_final.pt",
    )
    print("Pretraining completed successfully! Final checkpoint saved to checkpoints/checkpoint_supervised_final.pt")


if __name__ == "__main__":
    train()
