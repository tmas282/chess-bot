import chess
import numpy as np
import torch

from utils import get_device


class ChessEnvironment:
    CLIP_THRESHOLD = 200

    @staticmethod
    def board_to_tensor(
        board: chess.Board,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Encodes board state into a (17, 8, 8) tensor via early fusion:
          - Planes 0-5:   White pieces (P, N, B, R, Q, K)
          - Planes 6-11:  Black pieces (P, N, B, R, Q, K)
          - Plane 12:     Turn (1.0 for White, 0.0 for Black)
          - Plane 13:     White Queenside castling rights (1.0 or 0.0)
          - Plane 14:     White Kingside castling rights (1.0 or 0.0)
          - Plane 15:     Black Queenside castling rights (1.0 or 0.0)
          - Plane 16:     Black Kingside castling rights (1.0 or 0.0)
        """
        if device is None:
            device = get_device()

        planes = np.zeros((17, 64), dtype=np.float32)

        for piece_type in range(1, 7):
            for sq in board.pieces(piece_type, chess.WHITE):
                planes[piece_type - 1, sq] = 1.0
            for sq in board.pieces(piece_type, chess.BLACK):
                planes[piece_type + 5, sq] = 1.0

        if board.turn == chess.WHITE:
            planes[12, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[13, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[14, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[15, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[16, :] = 1.0

        planes = np.reshape(planes, shape=(17, 8, 8))
        return torch.tensor(planes, device=device, dtype=dtype)

    def __init__(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        fen: str | None = chess.STARTING_FEN,
    ):
        self.device = get_device() if device is None else torch.device(device)
        self.dtype = dtype
        self.board = chess.Board(fen=fen)
        self.steps = 0

    def reset(self, fen: str | None = chess.STARTING_FEN) -> torch.Tensor:
        self.board.set_fen(fen) if fen else self.board.reset()
        self.steps = 0
        return self.get_state_tensor()

    def get_state_tensor(self) -> torch.Tensor:
        return ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype)

    def get_legal_moves(self) -> list[chess.Move]:
        return list(self.board.legal_moves)

    def get_outcome(self, claim_draw: bool = True) -> tuple[bool, float]:
        """
        Checks terminal outcome from White's perspective (+1.0 White win, -1.0 Black win, 0.0 draw).
        Returns: (is_terminal, white_centric_value)
        """
        outcome = self.board.outcome(claim_draw=claim_draw)
        if outcome is not None:
            if outcome.winner == chess.WHITE:
                return True, 1.0
            elif outcome.winner == chess.BLACK:
                return True, -1.0
            else:
                return True, 0.0
        return False, 0.0

    def step(
        self,
        move: chess.Move,
        player_side: chess.Color = chess.WHITE,
    ) -> tuple[torch.Tensor, float, bool, bool]:
        """
        Applies a legal chess.Move to the environment.
        Returns: (next_state_tensor, reward, terminated, truncated)
        """
        if not self.board.is_legal(move):
            penalty = -1.0 if self.board.turn == player_side else 1.0
            return self.get_state_tensor(), penalty, True, False

        self.board.push(move)
        self.steps += 1

        is_term, white_val = self.get_outcome(claim_draw=True)
        truncated = (self.steps >= self.CLIP_THRESHOLD) and not is_term

        if is_term:
            reward = white_val if player_side == chess.WHITE else -white_val
            terminated = True
        else:
            reward = 0.0
            terminated = False

        return self.get_state_tensor(), reward, terminated, truncated
