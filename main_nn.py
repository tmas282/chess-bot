import sys
from heuristic_nn import use_model

if __name__ == "__main__":
    for fen in sys.stdin:
        fen = fen.strip()
        if not fen:
            continue
        y = use_model(fen)
        print(y)