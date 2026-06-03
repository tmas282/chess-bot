from abc import ABC, abstractmethod

import numpy as np


class Piece(ABC):
    @abstractmethod
    def get_position(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def is_white(self) -> bool:
        pass

    @abstractmethod
    def get_available_moves(self, board) -> np.ndarray:
        pass

    @abstractmethod
    def get_value(self) -> np.ndarray:
        pass


class Rook(Piece):
    def get_position(self) -> tuple[int, int]:
        return self.position

    def is_white(self) -> bool:
        return self.white

    def get_value(self) -> np.ndarray:
        return np.array(5 * 1 if self.white else -1)

    def get_available_moves(self, board) -> np.ndarray:
        available_moves = []
        y_temp = 0
        while True:
            y_temp += 1
            possible_y = self.get_position()[1] + y_temp
            if possible_y > 7:
                break
            available_moves.append((self.get_position()[0], possible_y))

        y_temp = 0
        while True:
            y_temp += 1
            possible_y = self.get_position()[1] - y_temp
            if possible_y < 0:
                break
            available_moves.append((self.get_position()[0], possible_y))

        x_temp = 0
        while True:
            x_temp += 1
            possible_x = self.get_position()[0] + x_temp
            if possible_x > 7:
                break
            available_moves.append((possible_x, self.get_position()[1]))

        x_temp = 0
        while True:
            x_temp += 1
            possible_x = self.get_position()[0] - x_temp
            if possible_x < 0:
                break
            available_moves.append((possible_x, self.get_position()[1]))

        return np.array(available_moves)

    def __init__(self, position: tuple[int, int], white=True) -> None:
        self.white = white
        self.position = position


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
