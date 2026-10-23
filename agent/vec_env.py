import multiprocessing as mp

import torch

from agent.environment import ChessEnvironment


def _worker_fn(remote, parent_remote):
    """Worker process that manages a single independent ChessEnvironment."""
    parent_remote.close()
    env = ChessEnvironment(device="cpu", dtype=torch.bfloat16)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                state = env.reset()
                mask = env.get_action_mask(env.board, device="cpu")
                remote.send((state, mask))
            elif cmd == "step":
                action, is_white = data
                next_state, reward, terminated, truncated = env.step(action, is_white)
                done = terminated or truncated
                mask = None
                if not done:
                    mask = env.get_action_mask(env.board, device="cpu")
                remote.send((next_state, reward, terminated, truncated, mask))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Command '{cmd}' not recognized.")
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass


class SubprocVectorChessEnv:
    """
    Runs multiple ChessEnvironments in separate CPU processes using multiprocessing Pipes.
    Bypasses the Python GIL for parallel legal move generation and board representation.
    """
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = []

        ctx = mp.get_context("spawn")
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = ctx.Process(target=_worker_fn, args=(work_remote, remote), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset_all(self):
        """Resets all environments and returns stacked initial states and masks."""
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        states, masks = zip(*results)
        return torch.stack(states), torch.stack(masks)

    def reset_one(self, env_idx: int):
        """Resets a single environment upon game termination."""
        self.remotes[env_idx].send(("reset", None))
        return self.remotes[env_idx].recv()

    def step(self, actions, is_whites):
        """Steps all environments in parallel with their corresponding actions."""
        for remote, action, is_white in zip(self.remotes, actions, is_whites):
            remote.send(("step", (action, is_white)))
        results = [remote.recv() for remote in self.remotes]
        # next_states, rewards, terminateds, truncateds, masks
        next_states, rewards, terminateds, truncateds, masks = zip(*results)
        return torch.stack(next_states), rewards, terminateds, truncateds, masks

    def close(self):
        """Immediately and safely terminates all worker processes."""
        for p in self.processes:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception as e:
                print(f"Exception {e}")
        for remote in self.remotes:
            try:
                remote.close()
            except Exception as e:
                print(f"Exception {e}")
