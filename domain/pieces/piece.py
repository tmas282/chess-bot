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
