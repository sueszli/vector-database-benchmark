import array
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Tuple
import numpy as np
from .window import CRWindow, Window
Elem = Tuple[int, int]

class CostMatrix(ABC):
    """
    (n+1) x (m+1) Matrix
    Cell (i,j) corresponds to minimum total cost/distance of matching elements (i-1, j-1) in series1 and series2.

    Row 0 and column 0, are typically set to infinity, to prevent matching before the first element
    """
    n: int
    m: int

    @abstractmethod
    def fill(self, value: float):
        if False:
            return 10
        pass

    @abstractmethod
    def __getitem__(self, item):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def __iter__(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def to_dense(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Returns\n        -------\n        Dense n x m numpy array, where empty cells are set to np.inf\n        '
        pass

    @staticmethod
    def _from_window(window: Window):
        if False:
            i = 10
            return i + 15
        '\n        Creates a cost matrix from a window.\n        Depending on the density of the active cells in the window,\n        will select either a dense or sparse storage representation.\n\n        Parameters\n        ----------\n        window\n            Takes a `Window` defining which cells are active and which are empty\n\n        Returns\n        -------\n        CostMatrix\n        '
        density = len(window) / ((window.n + 1) * (window.m + 1))
        if isinstance(window, CRWindow) and density < 0.5:
            return SparseCostMatrix(window)
        else:
            return DenseCostMatrix(window.n, window.m)

class DenseCostMatrix(np.ndarray, CostMatrix):

    def __new__(self, n, m):
        if False:
            while True:
                i = 10
        self.n = n
        self.m = m
        return super().__new__(self, (n + 1, m + 1), float)

    def to_dense(self) -> np.ndarray:
        if False:
            print('Hello World!')
        return self[1:, 1:]

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for n in range(1, self.n):
            for m in range(1, self.m):
                yield (n, m)

class SparseCostMatrix(CostMatrix):

    def __init__(self, window: CRWindow):
        if False:
            print('Hello World!')
        self.n = window.n
        self.m = window.m
        self.window = window
        self.offsets = np.empty(self.n + 2, dtype=int)
        self.column_ranges = window.column_ranges
        self.offsets[0] = 0
        np.cumsum(window.column_lengths(), out=self.offsets[1:])
        len = self.offsets[-1]
        self.offsets = array.array('i', self.offsets)
        self.dense = array.array('f', repeat(np.inf, len))

    def fill(self, value):
        if False:
            print('Hello World!')
        if value != np.inf:
            for i in range(len(self.dense)):
                self.dense[i] = value

    def to_dense(self) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        matrix = np.empty((self.n, self.m))
        matrix.fill(np.inf)
        if isinstance(self.window, CRWindow):
            ranges = self.window.column_ranges
            lengths = self.window.column_lengths()
            for i in range(1, self.n + 1):
                start = ranges[i * 2 + 0] - 1
                end = ranges[i * 2 + 1] - 1
                len = lengths[i]
                offset = self.offsets[i]
                matrix[i - 1][start:end] = self.dense[offset:offset + len]
        else:
            for i in range(1, self.n + 1):
                column_start = self.offsets[i]
                for j in range(1, self.m + 1):
                    column_idx = self.window.column_index(i)
                    if column_idx == -1:
                        continue
                    matrix[i - 1, j - 1] = self.dense[column_start + column_idx]
        return matrix

    def __getitem__(self, elem: Elem):
        if False:
            print('Hello World!')
        (i, j) = elem
        start = self.column_ranges[i * 2 + 0]
        end = self.column_ranges[i * 2 + 1]
        if start <= j < end:
            return self.dense[self.offsets[i] + j - start]
        return np.inf

    def __setitem__(self, elem, value):
        if False:
            for i in range(10):
                print('nop')
        (i, j) = elem
        start = self.column_ranges[i * 2 + 0]
        self.dense[self.offsets[i] + j - start] = value

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.window.__iter__()