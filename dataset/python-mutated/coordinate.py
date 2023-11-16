"""
A class to store a coordinate, used by the [DataTable][textual.widgets.DataTable].
"""
from __future__ import annotations
from typing import NamedTuple

class Coordinate(NamedTuple):
    """An object representing a row/column coordinate within a grid."""
    row: int
    'The row of the coordinate within a grid.'
    column: int
    'The column of the coordinate within a grid.'

    def left(self) -> Coordinate:
        if False:
            print('Hello World!')
        'Get the coordinate to the left.\n\n        Returns:\n            The coordinate to the left.\n        '
        (row, column) = self
        return Coordinate(row, column - 1)

    def right(self) -> Coordinate:
        if False:
            while True:
                i = 10
        'Get the coordinate to the right.\n\n        Returns:\n            The coordinate to the right.\n        '
        (row, column) = self
        return Coordinate(row, column + 1)

    def up(self) -> Coordinate:
        if False:
            print('Hello World!')
        'Get the coordinate above.\n\n        Returns:\n            The coordinate above.\n        '
        (row, column) = self
        return Coordinate(row - 1, column)

    def down(self) -> Coordinate:
        if False:
            i = 10
            return i + 15
        'Get the coordinate below.\n\n        Returns:\n            The coordinate below.\n        '
        (row, column) = self
        return Coordinate(row + 1, column)