from random import random

import chess
import numpy as np


def heuristic(fen: str):
    return random() * 20 - 10


def minimax(board_fen: str, depth=3, my_turn=True):
    board = chess.Board(fen=board_fen)
    legal = np.array(list(board.legal_moves))
    if depth == 0 or len(legal) == 0:
        return heuristic(board_fen)
    evaluations = []
    for move in list(board.legal_moves):
        board.push(move)
        evaluations.append(minimax(board.fen(), depth=depth - 1, my_turn=not my_turn))
        board.pop()
    if my_turn:
        return max(evaluations)
    else:
        return min(evaluations)


def evaluate_positions(board_fen: str, depth=3):
    board = chess.Board(fen=board_fen)
    evaluations = []

    for move in np.array(list(board.legal_moves)):
        board.push(move)
        evaluations.append(minimax(board.fen(), depth=depth - 1, my_turn=False))
        board.pop()
    return list(board.legal_moves)[evaluations.index(max(evaluations))]
