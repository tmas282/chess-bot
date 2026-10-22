import chess
import numpy as np
import torch

# -------------------------------------------------------------
# Action Space: 4164 actions
#   - Indices [0 .. 4031]: Standard moves (and Queen promotions)
#   - Indices [4032 .. 4163]: Underpromotions (Knight, Bishop, Rook)
# -------------------------------------------------------------
UNDERPROMOTIONS = []
for color in [chess.WHITE, chess.BLACK]:
    from_rank = 6 if color == chess.WHITE else 1
    to_rank = 7 if color == chess.WHITE else 0
    for file in range(8):
        f = chess.square(file, from_rank)
        for dest_file in [file - 1, file, file + 1]:
            if 0 <= dest_file <= 7:
                t = chess.square(dest_file, to_rank)
                for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                    UNDERPROMOTIONS.append(chess.Move(f, t, promotion=piece))

ACTION_TO_UNDERPROMO = {4032 + i: move for i, move in enumerate(UNDERPROMOTIONS)}
UNDERPROMO_TO_ACTION = {move: 4032 + i for i, move in enumerate(UNDERPROMOTIONS)}
TOTAL_ACTIONS = 4032 + len(UNDERPROMOTIONS)  # 4164

class ChessEnvironment:
    NUMBER_OF_ACTIONS = TOTAL_ACTIONS  # 4164
    CLIP_THRESHOLD = 128
    @staticmethod
    def action_to_move(action: int, board: chess.Board | None = None) -> chess.Move:
        """
        Converts an action index (0 <= action < 4164) to a chess.Move.
        """
        if not (0 <= action < TOTAL_ACTIONS):
            raise ValueError(f"Action {action} out of bounds (0 <= action < {TOTAL_ACTIONS})")

        # 1. Check underpromotions [4032 .. 4163]
        if action >= 4032:
            return ACTION_TO_UNDERPROMO[int(action)]

        # 2. Standard moves [0 .. 4031]
        from_square = action // 63
        to_offset = action % 63
        to_square = to_offset if to_offset < from_square else to_offset + 1

        move = chess.Move(from_square, to_square)

        # If it's an 8th/1st rank pawn advance, default base action promotes to Queen
        if board is not None and move not in board.legal_moves:
            queen_promo = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if queen_promo in board.legal_moves:
                return queen_promo

        return move

    @staticmethod
    def move_to_action(move: chess.Move) -> int:
        """
        Converts a chess.Move to its unique action index in [0 .. 4163].
        """
        # If it's an underpromotion (Knight, Bishop, Rook)
        if move.promotion is not None and move.promotion != chess.QUEEN:
            return UNDERPROMO_TO_ACTION[move]

        # Standard move or Queen promotion
        from_sq = move.from_square
        to_sq = move.to_square
        offset = to_sq if to_sq < from_sq else to_sq - 1
        return from_sq * 63 + offset

    @staticmethod
    def get_action_mask(board: chess.Board, device: torch.device | str = "cpu") -> torch.Tensor:
        """
        Returns a boolean mask of shape (4164,) with True for every legal move.
        """
        mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.bool, device=device)
        for move in board.legal_moves:
            action_idx = ChessEnvironment.move_to_action(move)
            mask[action_idx] = True
        return mask
        
    @staticmethod
    def board_to_tensor(board: chess.Board, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
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
        planes = np.zeros((17, 64), dtype=np.float32)

        # 12 piece planes
        for piece_type in range(1, 7):
            for sq in board.pieces(piece_type, chess.WHITE):
                planes[piece_type - 1, sq] = 1.0

            for sq in board.pieces(piece_type, chess.BLACK):
                planes[piece_type + 5, sq] = 1.0

        # Broadcast metadata across full 8x8 planes (64 squares)
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

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self, fen: str | None = chess.STARTING_FEN):
        self.board = chess.Board(fen=fen)
        self.steps = 0
        return ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype)

    def step(self, action, player_side=chess.WHITE):
        action = ChessEnvironment.action_to_move(action, self.board)
        if not self.board.is_legal(action):
            out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), -1 if self.board.turn == chess.WHITE else 1, True, False
            return out
        self.board.push(action)
        self.steps += 1
        g_outcome = self.board.outcome()
        out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), 0, False, False
        if g_outcome is None:
            pass
        elif g_outcome.winner == chess.WHITE:
            out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), 1 if player_side == chess.WHITE else -1, True, False
        elif g_outcome.winner == chess.BLACK:
            out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), -1 if player_side == chess.WHITE else 1, True, False
        elif g_outcome.winner is None:
            out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), 0, True, False
        elif self.steps == ChessEnvironment.CLIP_THRESHOLD:
            out = ChessEnvironment.board_to_tensor(self.board, self.device, self.dtype), 0, False, True
        return out
