import time

import numpy as np
import torch

from agent import actor_vit, critic_vit
from agent.environment import ChessEnvironment
from agent.loss_func import A2CombinedLoss
from utils import plot_training_metrics, save_checkpoint

LEARNING_RATE = 1e-5
GAMMA = 0.99
ENTROPY_COEFF = 0.01
BETA_CRITIC_WEIGHT = 1.0
NUM_EPISODES = 4_000_000
EPISODE_TRAIN_THRESHOLD = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    actor = actor_vit.ActorViT(embed_dim=256, depth=4).to(DEVICE).to(torch.bfloat16)

    critic = critic_vit.CriticViT(embed_dim=256, depth=4).to(DEVICE).to(torch.bfloat16)

    optim = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        weight_decay=0.01, lr=LEARNING_RATE)
    a2c_loss_fn = A2CombinedLoss(ENTROPY_COEFF, BETA_CRITIC_WEIGHT).to(DEVICE).to(torch.bfloat16)

    all_rewards = []
    a2c_loss_info = []
    batch_start_time = time.perf_counter()
    batch_steps = 0

    env = ChessEnvironment(device=DEVICE, dtype=torch.bfloat16)

    states = []
    masks = []
    actions = []
    batch_qvals = []
    
    for episode in range(NUM_EPISODES + 1):
        state = env.reset()
        
        terminated = False
        truncated = False
        is_white = True

        rewards = []
        Qvals = []
        next_state: torch.Tensor | None = None
        with torch.no_grad():
            while not terminated and not truncated:
                logits = actor(torch.unsqueeze(state, 0)).squeeze(0)
    
                mask = env.get_action_mask(env.board, device=DEVICE)
                masked_logits = logits.masked_fill(~mask, -1e2)
    
                actions_distribution = torch.distributions.Categorical(logits=masked_logits)
                random_action = actions_distribution.sample()
    
                next_state, reward, terminated, truncated = env.step(random_action, is_white)

                rewards.append(reward)
                states.append(state)
                actions.append(random_action)
                masks.append(mask)
    
                is_white = not is_white
                state = next_state

                batch_steps += 1
                

            all_rewards.append(np.sum(rewards))
            now = time.perf_counter()
            elapsed_batch = now - batch_start_time
            batch_sps = batch_steps / elapsed_batch
            print(f"Ep {episode}: Reward {np.sum(rewards)} / Steps Per Second: {batch_sps:.1f}")
            
            Qval = 0.0 if (terminated and not truncated) else critic(next_state).item()
            Qvals = []
            for r in reversed(rewards):
                Qval = r + GAMMA * Qval
                Qvals.insert(0, Qval)
            batch_qvals.extend(Qvals)

        if episode % EPISODE_TRAIN_THRESHOLD != 0:
            continue
        batch_states = torch.stack(states).to(DEVICE).to(torch.bfloat16)
        batch_actions = torch.tensor(actions).to(DEVICE).to(torch.bfloat16)
        batch_masks = torch.stack(masks).to(DEVICE).to(torch.bool)
        
        Qvals = torch.Tensor(batch_qvals).to(DEVICE).to(torch.bfloat16).unsqueeze(1)

        actor.train()
        critic.train()

        logits = actor(batch_states)
        values = critic(batch_states)

        masked_logits = logits.masked_fill(~batch_masks, -1e2)
        dist = torch.distributions.Categorical(logits=masked_logits)
        batch_actions_logs_probs = dist.log_prob(batch_actions)

        a2c_loss = a2c_loss_fn(batch_actions_logs_probs, Qvals, values, dist.entropy().sum())

        optim.zero_grad(set_to_none=True)
        a2c_loss.backward()
        optim.step()

        states = []
        masks = []
        actions = []
        batch_qvals = []

        batch_start_time = time.perf_counter()
        batch_steps = 0
        print(f"Episode {episode}: A2C loss: {float(a2c_loss)}")
        a2c_loss_info.append(float(a2c_loss))
        save_checkpoint(
            episode=episode,
            actor=actor,
            critic=critic,
            optimizer=optim,
            loss=float(a2c_loss),
            save_dir="checkpoints"
        )
        plot_training_metrics(all_rewards, a2c_loss_info)

if __name__ == "__main__":
    train()
