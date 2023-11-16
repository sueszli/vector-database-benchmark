"""Base plotter API."""
from abc import ABC, abstractmethod
from typing import Any
from qiskit.visualization.timeline import core

class BasePlotter(ABC):
    """Base class of Qiskit plotter."""

    def __init__(self, canvas: core.DrawerCanvas):
        if False:
            for i in range(10):
                print('nop')
        'Create new plotter.\n        Args:\n            canvas: Configured drawer canvas object.\n        '
        self.canvas = canvas

    @abstractmethod
    def initialize_canvas(self):
        if False:
            while True:
                i = 10
        'Format appearance of the canvas.'
        raise NotImplementedError

    @abstractmethod
    def draw(self):
        if False:
            while True:
                i = 10
        'Output drawings stored in canvas object.'
        raise NotImplementedError

    @abstractmethod
    def save_file(self, filename: str):
        if False:
            print('Hello World!')
        'Save image to file.\n        Args:\n            filename: File path to output image data.\n        '
        raise NotImplementedError

    @abstractmethod
    def get_image(self, interactive: bool=False) -> Any:
        if False:
            i = 10
            return i + 15
        'Get image data to return.\n        Args:\n            interactive: When set `True` show the circuit in a new window.\n                This depends on the matplotlib backend being used supporting this.\n        Returns:\n            Image data. This depends on the plotter API.\n        '
        raise NotImplementedError