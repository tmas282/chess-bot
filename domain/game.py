from numpy import array

from domain.pieces import Bishop, Piece, Rook


class Game:
    def get_possible_moves(self, is_white=True):
        for i in self.board:
            if isinstance(i, Piece) and i.is_white() == is_white:
                print(i.get_available_moves(self.board))

    def __init__(self):
        self.board = array([Rook((3, 3), True)])
