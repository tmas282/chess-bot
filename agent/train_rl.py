import time

import torch

from agent import actor_vit, critic_vit
from agent.loss_func import A2CombinedLoss
from agent.vec_env import SubprocVectorChessEnv
from utils import plot_training_metrics, save_checkpoint

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
ENTROPY_COEFF = 0.01
BETA_CRITIC_WEIGHT = 1.0
NUM_ENVS = 4 # Number of parallel environments running on CPU cores
ROLLOUT_STEPS = 128 # Steps per env before training (Batch size = 4 * 128 = 512)
TOTAL_UPDATES = 1_000_000
# Train 4 million games Math:
# TOTAL_UPDATES * NUM_ENVS
# Obs: assuming one environment is roughly one game
# 
# For a good CPU+GPU:
# TOTAL_UPDATES = 500_000
# NUM_ENVS = 16
# Batch size is 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_THRESHOLD = 500

def train():
    actor = actor_vit.ActorViT(embed_dim=256, depth=4).to(DEVICE).to(torch.bfloat16)
    critic = critic_vit.CriticViT(embed_dim=256, depth=4).to(DEVICE).to(torch.bfloat16)

    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        weight_decay=0.01,
        lr=LEARNING_RATE,
    )
    a2c_loss_fn = A2CombinedLoss(ENTROPY_COEFF, BETA_CRITIC_WEIGHT).to(DEVICE).to(torch.bfloat16)

    all_rewards = []
    a2c_loss_info = []
    batch_start_time = time.perf_counter()
    batch_steps = 0

    print(f"Starting {NUM_ENVS} parallel environments across CPU cores...")
    vec_env = SubprocVectorChessEnv(num_envs=NUM_ENVS)

    states, masks = vec_env.reset_all()
    states = states.to(DEVICE).to(torch.bfloat16)
    masks = masks.to(DEVICE).to(torch.bool)
    is_whites = [True] * NUM_ENVS

    try:
        for update in range(1, TOTAL_UPDATES + 1):
            batch_states = []
            batch_actions = []
            batch_masks = []
            batch_rewards = []
            batch_dones = []

            # Rollout Phase
            with torch.no_grad():
                for _ in range(ROLLOUT_STEPS):
                    logits = actor(states)
                    masked_logits = logits.masked_fill(~masks, -1e2)

                    actions_distribution = torch.distributions.Categorical(logits=masked_logits)
                    actions = actions_distribution.sample()

                    next_states, rewards, terms, truncs, next_masks = vec_env.step(
                        actions.tolist(), is_whites
                    )

                    dones = [t or tr for t, tr in zip(terms, truncs)]

                    batch_states.append(states)
                    batch_actions.append(actions)
                    batch_masks.append(masks)
                    batch_rewards.append(torch.tensor(rewards, device=DEVICE, dtype=torch.bfloat16))
                    batch_dones.append(torch.tensor(dones, device=DEVICE, dtype=torch.bool))

                    # Auto-reset any finished games
                    next_states_list = []
                    next_masks_list = []
                    for i in range(NUM_ENVS):
                        if dones[i]:
                            all_rewards.append(rewards[i])
                            st, mk = vec_env.reset_one(i)
                            next_states_list.append(st)
                            next_masks_list.append(mk)
                            is_whites[i] = True
                        else:
                            next_states_list.append(next_states[i])
                            next_masks_list.append(next_masks[i])
                            is_whites[i] = not is_whites[i]

                    states = torch.stack(next_states_list).to(DEVICE).to(torch.bfloat16)
                    masks = torch.stack(next_masks_list).to(DEVICE).to(torch.bool)
                    batch_steps += NUM_ENVS

                # Compute Qvals backwards across all parallel envs
                running_q = critic(states).squeeze(-1)
                qvals = torch.zeros((ROLLOUT_STEPS, NUM_ENVS), device=DEVICE, dtype=torch.bfloat16)
                for t in reversed(range(ROLLOUT_STEPS)):
                    running_q = batch_rewards[t] + GAMMA * running_q * (~batch_dones[t])
                    qvals[t] = running_q

            # Training Phase
            
            batch_states_t = torch.cat(batch_states, dim=0)
            batch_actions_t = torch.cat(batch_actions, dim=0)
            batch_masks_t = torch.cat(batch_masks, dim=0)
            Qvals = qvals.view(-1, 1)

            actor.train()
            critic.train()

            logits = actor(batch_states_t)
            values = critic(batch_states_t)

            masked_logits = logits.masked_fill(~batch_masks_t, -1e2)
            dist = torch.distributions.Categorical(logits=masked_logits)
            batch_actions_logs_probs = dist.log_prob(batch_actions_t)

            a2c_loss = a2c_loss_fn(batch_actions_logs_probs, Qvals, values, dist.entropy().sum())

            optim.zero_grad(set_to_none=True)
            a2c_loss.backward()
            optim.step()

            # Logging & Checkpointing
            now = time.perf_counter()
            elapsed_batch = now - batch_start_time
            batch_sps = batch_steps / elapsed_batch if elapsed_batch > 0 else 0
            batch_start_time = time.perf_counter()
            batch_steps = 0

            print(f"Update {update}: A2C loss: {float(a2c_loss):.4f} / Steps Per Second: {batch_sps:.1f}")
            a2c_loss_info.append(float(a2c_loss))

            if update % CHECKPOINT_THRESHOLD != 0:
                continue
            save_checkpoint(
                episode=update,
                actor=actor,
                critic=critic,
                optimizer=optim,
                loss=float(a2c_loss),
                save_dir="checkpoints",
            )

    except KeyboardInterrupt:
        print("\n[!] Training stopped via Ctrl+C. Saving safety checkpoint...")
        save_checkpoint(
            episode=update if "update" in locals() else 0,
            actor=actor,
            critic=critic,
            optimizer=optim,
            loss=float(a2c_loss) if "a2c_loss" in locals() else None,
            save_dir="checkpoints",
            filename="checkpoint_interrupted.pt",
        )
    finally:
        print("\n[Shutdown] Cleaning up worker processes...")
        vec_env.close()
        if len(a2c_loss_info) > 0:
            print("Plotting training metrics...")
            try:
                plot_training_metrics(all_rewards, a2c_loss_info, save_path="training_curves.png")
            except Exception as e:
                print(f"[ERROR] Failed to generate plot: {e}")
        else:
            print("No updates completed yet, nothing to plot.")


if __name__ == "__main__":
    train()
