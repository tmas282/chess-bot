import collections
import os
import random
import time

import chess
import torch
from torch import nn

from agent.environment import ChessEnvironment
from agent.value_agent import ValueIterationAgent
from agent.value_vit import ValueViT
from utils import get_device, load_checkpoint, plot_training_metrics, save_checkpoint

# Hyperparameters
# Total transitions collected = TOTAL_UPDATES * ROLLOUT_STEPS_PER_UPDATE * NUM_PARALLEL_GAMES
# Approximate games played = (Total transitions) / ~60 moves per game
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
GAMMA = 0.99
NUM_PARALLEL_GAMES = 4
REPLAY_BUFFER_CAPACITY = 10_000
MIN_REPLAY_SIZE = 512
BATCH_SIZE = 512
ROLLOUT_STEPS_PER_UPDATE = 200
TOTAL_UPDATES = 600_000
CHECKPOINT_INTERVAL = 250
LOG_INTERVAL = 10
DEVICE = get_device()


def train():
    print(f"Initializing ValueViT model on {DEVICE} (bfloat16)...")
    value_vit = ValueViT().to(DEVICE).to(torch.bfloat16)

    supervised_checkpoint = os.path.join("checkpoints", "checkpoint_supervised_final.pt")
    if os.path.isfile(supervised_checkpoint):
        print(f"Loading pretrained supervised weights from: {supervised_checkpoint}")
        load_checkpoint(supervised_checkpoint, value_net=value_vit, device=DEVICE)

    agent = ValueIterationAgent(model=value_vit, device=DEVICE, dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        value_vit.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.MSELoss()

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_CAPACITY)

    envs = [ChessEnvironment(device="cpu", dtype=torch.bfloat16) for _ in range(NUM_PARALLEL_GAMES)]
    trajectories: list[list[tuple[torch.Tensor, chess.Color]]] = [[] for _ in range(NUM_PARALLEL_GAMES)]

    all_rewards = []
    value_losses = []
    white_wins = 0
    black_wins = 0
    draws = 0
    total_games = 0
    update = 0
    loss_val = None

    print(f"Starting {NUM_PARALLEL_GAMES} parallel self-play environments for Value Iteration RL...")
    batch_start_time = time.perf_counter()
    collected_steps = 0

    try:
        for update in range(1, TOTAL_UPDATES + 1):
            # 1. Self-Play Rollout Phase
            value_vit.train()  # Keep train mode enabled so NoisyLinear samples parameter noise
            for _ in range(ROLLOUT_STEPS_PER_UPDATE):
                # Record pre-transition state tensors
                for i, env in enumerate(envs):
                    trajectories[i].append((env.get_state_tensor(), env.board.turn))

                # Batch action selection via ValueIterationAgent
                actions = agent.select_actions_batch(envs)

                # Step environments
                for i, env in enumerate(envs):
                    _, reward, terminated, truncated = env.step(actions[i])
                    collected_steps += 1

                    if terminated or truncated:
                        if terminated:
                            outcome_z = reward
                        else:
                            outcome_z = agent.evaluate_state(env)

                        # Backward Bellman target assignment: y_t = z * gamma^(T - t)
                        traj = trajectories[i]
                        running_target = outcome_z
                        for t in reversed(range(len(traj))):
                            st_tensor, _ = traj[t]
                            replay_buffer.append((st_tensor, running_target))
                            running_target *= GAMMA

                        all_rewards.append(outcome_z)
                        total_games += 1
                        if outcome_z > 0.5:
                            white_wins += 1
                        elif outcome_z < -0.5:
                            black_wins += 1
                        else:
                            draws += 1

                        env.reset()
                        trajectories[i] = []

            # Value Iteration
            if len(replay_buffer) < MIN_REPLAY_SIZE:
                continue

            samples = random.sample(replay_buffer, BATCH_SIZE)
            states_batch = torch.stack([s[0] for s in samples]).to(DEVICE).to(torch.bfloat16)
            targets_batch = (
                torch.tensor([s[1] for s in samples], device=DEVICE, dtype=torch.bfloat16)
                .unsqueeze(1)
            )

            value_vit.train()
            value_vit.reset_noise()
            preds = value_vit(states_batch)
            loss = loss_fn(preds, targets_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_vit.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = float(loss.item())
            value_losses.append(loss_val)

            # Logging
            if update % LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - batch_start_time
                sps = collected_steps / elapsed if elapsed > 0 else 0
                batch_start_time = time.perf_counter()
                collected_steps = 0

                print(
                    f"Update {update:6d} | Loss: {loss_val:.4f} | Games: {total_games:4d} "
                    f"(W:{white_wins} B:{black_wins} D:{draws}) | Steps/s: {sps:5.1f} | Buffer: {len(replay_buffer):5d}"
                )

            if update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    episode=update,
                    value_net=value_vit,
                    optimizer=optimizer,
                    loss=loss_val,
                    save_dir="checkpoints",
                )

    except KeyboardInterrupt:
        print("\n[!] Training stopped via KeyboardInterrupt. Saving checkpoint...")
        save_checkpoint(
            episode=update,
            value_net=value_vit,
            optimizer=optimizer,
            loss=loss_val,
            save_dir="checkpoints",
            filename="checkpoint_interrupted.pt",
        )
    finally:
        print("\nTraining session ended.")
        if len(value_losses) > 0:
            print("Plotting training metrics...")
            try:
                plot_training_metrics(all_rewards, value_losses, save_path="training_curves.png")
            except Exception as e:
                print(f"[ERROR] Failed to generate plot: {e}")
        else:
            print("No updates completed yet, nothing to plot.")


if __name__ == "__main__":
    train()
