from random import random

import chess
import numpy as np


def heuristic(fen: str):
    return random() * 20 - 10


def minimax_computation(legal_move, board_fen: str, depth=3, my_turn=True, max_play_is_white=True):
    if depth == 0:
        return heuristic(board_fen)
    board = chess.Board(fen=board_fen)
    board.push(legal_move)
    res = minimax(board.fen(), depth=depth - 1, my_turn=not my_turn, max_play_is_white=max_play_is_white)
    board.pop()
    del board
    return res


def minimax(board_fen: str, depth=3, my_turn=True, max_play_is_white=True):
    board = chess.Board(fen=board_fen)
    legal = np.array(list(board.legal_moves))
    if depth == 0 or len(legal) == 0:
        return heuristic(board_fen)
    vectorized_computation = np.vectorize(minimax_computation)
    evaluations = vectorized_computation(legal, board.fen(), depth - 1, not my_turn)
    if my_turn:
        return np.max(evaluations) if max_play_is_white else np.min(evaluations)
    else:
        return np.min(evaluations) if max_play_is_white else np.max(evaluations)


def evaluate_positions(board_fen: str, depth=3, max_play_is_white=True):
    board = chess.Board(fen=board_fen)
    legal_moves = np.array(list(board.legal_moves))
    evaluations = []
    vectorized_computation = np.vectorize(minimax_computation)
    evaluations = vectorized_computation(
        legal_moves, board.fen(), depth - 1, my_turn=False
    )
    return legal_moves[np.argmax(evaluations)] if max_play_is_white else legal_moves[np.argmin(evaluations)]
