import chess
import torch
from torch import nn

from agent.environment import ChessEnvironment
from agent.value_vit import ValueViT
from utils import get_device


class ValueIterationAgent:
    """
    Value Iteration Agent for Chess.
    Leverages the environment's known deterministic transition model to evaluate
    successor states directly via a learned ValueViT network with NoisyLinear exploration.

    Value semantics (White-Centric):
      +1.0: Guaranteed / Expected White victory
       0.0: Draw / Equal position
      -1.0: Guaranteed / Expected Black victory
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = get_device() if device is None else torch.device(device)
        self.dtype = dtype
        if model is None:
            self.model = ValueViT().to(self.device).to(self.dtype)
        else:
            self.model = model.to(self.device).to(self.dtype)
        self.model.eval()

    def reset_noise(self):
        if hasattr(self.model, "reset_noise") and isinstance(self.model, ValueViT):
            self.model.reset_noise()

    @torch.inference_mode()
    def evaluate_state(self, env: ChessEnvironment) -> float:
        """
        Evaluates the current state of the environment, returning its White-centric value in [-1.0, 1.0].
        """
        is_term, term_val = env.get_outcome(claim_draw=True)
        if is_term:
            return term_val

        tensor = env.get_state_tensor().to(self.device).unsqueeze(0)
        value = self.model(tensor).item()
        return float(value)

    @torch.inference_mode()
    def evaluate_successors(
        self,
        env: ChessEnvironment,
    ) -> tuple[list[chess.Move], torch.Tensor]:
        """
        Computes all legal moves and their deterministic successor state values for an environment.
        """
        batch_results = self.evaluate_successors_batch([env])
        return batch_results[0]

    @torch.inference_mode()
    def evaluate_successors_batch(
        self,
        envs: list[ChessEnvironment],
    ) -> list[tuple[list[chess.Move], torch.Tensor]]:
        """
        Batches successor evaluations across multiple environments (e.g., parallel self-play games).
        Queries each environment's legal moves and board, then aggregates all non-terminal
        successor states into a single GPU forward pass.
        """
        results: list[tuple[list[chess.Move], torch.Tensor]] = []
        all_board_moves: list[list[chess.Move]] = []
        all_board_values: list[torch.Tensor] = []

        mapping: list[tuple[int, int]] = []  # (env_idx, move_idx)
        non_term_tensors: list[torch.Tensor] = []

        for env_idx, env in enumerate(envs):
            moves = env.get_legal_moves()
            all_board_moves.append(moves)
            if not moves:
                all_board_values.append(torch.empty(0, device=self.device, dtype=torch.float32))
                continue

            vals = torch.zeros(len(moves), device=self.device, dtype=torch.float32)
            all_board_values.append(vals)

            board = env.board
            for m_idx, move in enumerate(moves):
                board.push(move)
                is_term, term_val = env.get_outcome()
                if is_term:
                    vals[m_idx] = term_val
                else:
                    mapping.append((env_idx, m_idx))
                    non_term_tensors.append(
                        env.get_state_tensor().to(self.device)
                    )
                board.pop()

        if non_term_tensors:
            batch = torch.stack(non_term_tensors)
            model_vals = self.model(batch).squeeze(-1).to(torch.float32)
            for (env_idx, m_idx), val in zip(mapping, model_vals):
                all_board_values[env_idx][m_idx] = val

        for moves, vals in zip(all_board_moves, all_board_values):
            results.append((moves, vals))

        return results

    def select_action(
        self,
        env: ChessEnvironment,
    ) -> chess.Move:
        """
        Selects a legal move for a single environment.
        """
        return self.select_actions_batch([env])[0]

    def select_actions_batch(
        self,
        envs: list[ChessEnvironment],
    ) -> list[chess.Move]:
        """
        Selects legal moves for a batch of environments concurrently:
          - White maximizes White-centric successor value V(s')
          - Black minimizes White-centric successor value V(s')
        Exploration is handled entirely via parameter noise in NoisyLinear.
        """
        self.reset_noise()
        successors_batch = self.evaluate_successors_batch(envs)
        selected_moves: list[chess.Move] = []

        for env, (moves, values) in zip(envs, successors_batch):
            if not moves:
                raise ValueError("No legal moves available in the current environment state.")

            is_white = (env.board.turn == chess.WHITE)
            best_idx = int(torch.argmax(values).item()) if is_white else int(torch.argmin(values).item())
            selected_moves.append(moves[best_idx])

        return selected_moves
