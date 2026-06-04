import chess

from minimax import evaluate_positions

if __name__ == "__main__":
    board = chess.Board()
    while board.outcome() is None:
        print("Bot playing...")
        board.push(evaluate_positions(board.fen(), 5))
        print(f"Bot played. Current FEN: {board.fen()}")
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
