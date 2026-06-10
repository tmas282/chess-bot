import sys

import chess

from minimax import evaluate_positions


def start_game():
    sys.setrecursionlimit(10_000_000)
    board = chess.Board()
    player_is_white = input("Are you going to play as white? (type w if so): ") == 'w'
    is_player_turn = player_is_white
    while board.outcome() is None:
        if is_player_turn:
            legal_moves = board.legal_moves
            mv = input("Your move: ")
            while True:
                try:
                    legal = chess.Move.from_uci(mv) in legal_moves
                    if legal:
                        board.push(chess.Move.from_uci(mv))
                        break
                except Exception as e:
                    print(e)
                    pass
                mv = input("Illegal. Your move: ")
            is_player_turn = False
        else:
            print("Bot playing...")
            board.push(evaluate_positions(board.fen(), 7, max_play_is_white=not player_is_white))
            print(f"Bot played. Current FEN: {board.fen()}")
            is_player_turn = True

if __name__ == "__main__":
    start_game()