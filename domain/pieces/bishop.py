import numpy as np

from .piece import Piece


class Bishop(Piece):
    def get_position(self) -> tuple[int, int]:
        return self.position

    def is_white(self) -> bool:
        return self.white

    def get_value(self) -> np.ndarray:
        return np.array(3 * 1 if self.white else -1)

    def get_available_moves(self, board) -> np.ndarray:
        available_moves = []
        x_temp = 0
        y_temp = 0
        while True:
            x_temp += 1
            y_temp += 1
            possible_x = self.get_position()[0] + x_temp
            possible_y = self.get_position()[1] + y_temp
            if possible_x > 7 or possible_y > 7:
                break
            available_moves.append((possible_x, possible_y))

        x_temp = 0
        y_temp = 0
        while True:
            x_temp += 1
            y_temp += 1
            possible_x = self.get_position()[0] + x_temp
            possible_y = self.get_position()[1] - y_temp
            if possible_x > 7 or possible_y < 0:
                break
            available_moves.append((possible_x, possible_y))

        x_temp = 0
        y_temp = 0
        while True:
            x_temp += 1
            y_temp += 1
            possible_x = self.get_position()[0] - x_temp
            possible_y = self.get_position()[1] + y_temp
            if possible_x < 0 or possible_y > 7:
                break
            available_moves.append((possible_x, possible_y))

        x_temp = 0
        y_temp = 0
        while True:
            x_temp += 1
            y_temp += 1
            possible_x = self.get_position()[0] - x_temp
            possible_y = self.get_position()[1] - y_temp
            if possible_x < 0 or possible_y < 0:
                break
            available_moves.append((possible_x, possible_y))
        return np.array(available_moves)

    def __init__(self, position: tuple[int, int], white=True) -> None:
        self.white = white
        self.position = position
