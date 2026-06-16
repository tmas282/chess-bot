import numpy as np

def fens_to_arrays(fens: np.ndarray) -> np.ndarray:
    vec = np.vectorize(fen_to_array, signature="(1)->(14,8,8)")
    res = vec(fens)
    return res

def fen_to_array(fen: str | np.ndarray):
    if(type(fen) == np.ndarray):
        fen = fen[0]
    res = np.zeros(shape=(14,8,8),dtype=np.uint8)
    fen_split = fen.split(" ")

    if fen_split[1] == "w":
        res[12][0][0] = 1

    if fen_split[2] != "-":
        for i,v in enumerate("K", 'Q', 'k', 'q'):
            if fen_split[2].find(v):
                res[13][0][i] = 1

    for r, col_v in enumerate(fen_split[0].split("/")):
        c = 0
        for piece in col_v:
            try:
                n = int(piece)
                c += n
                continue
            except:
                pass
            if piece == "P":
                res[0][r][c] = 1
            if piece == "N":
                res[1][r][c] = 1
            if piece == "B":
                res[2][r][c] = 1
            if piece == "R":
                res[3][r][c] = 1
            if piece == "Q":
                res[4][r][c] = 1
            if piece == "K":
                res[5][r][c] = 1
            
            if piece == "p":
                res[6][r][c] = 1
            if piece == "n":
                res[7][r][c] = 1
            if piece == "b":
                res[8][r][c] = 1
            if piece == "r":
                res[9][r][c] = 1
            if piece == "q":
                res[10][r][c] = 1
            if piece == "k":
                res[11][r][c] = 1
            c += 1
    return res