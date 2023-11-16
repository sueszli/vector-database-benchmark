# -*- coding: utf-8 -*-
# vispy: gallery 2
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Control VisPy from Qt
=====================

Control a VisPy visualization with Qt-based (PyQt5) UI elements.

"""

import numpy as np
from PyQt5 import QtWidgets

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app

IMAGE_SHAPE = (600, 800)  # (height, width)
CANVAS_SIZE = (800, 600)  # (width, height)
NUM_LINE_POINTS = 200

COLORMAP_CHOICES = ["viridis", "reds", "blues"]
LINE_COLOR_CHOICES = ["black", "red", "blue"]


class MyMainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = CanvasWrapper()
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._connect_controls()

    def _connect_controls(self):
        self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
        self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)


class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()
        self.colormap_label = QtWidgets.QLabel("Image Colormap:")
        layout.addWidget(self.colormap_label)
        self.colormap_chooser = QtWidgets.QComboBox()
        self.colormap_chooser.addItems(COLORMAP_CHOICES)
        layout.addWidget(self.colormap_chooser)

        self.line_color_label = QtWidgets.QLabel("Line color:")
        layout.addWidget(self.line_color_label)
        self.line_color_chooser = QtWidgets.QComboBox()
        self.line_color_chooser.addItems(LINE_COLOR_CHOICES)
        layout.addWidget(self.line_color_chooser)

        layout.addStretch(1)
        self.setLayout(layout)


class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE)
        self.grid = self.canvas.central_widget.add_grid()

        self.view_top = self.grid.add_view(0, 0, bgcolor='cyan')
        image_data = _generate_random_image_data(IMAGE_SHAPE)
        self.image = visuals.Image(
            image_data,
            texture_format="auto",
            cmap=COLORMAP_CHOICES[0],
            parent=self.view_top.scene,
        )
        self.view_top.camera = "panzoom"
        self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)

        self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
        line_data = _generate_random_line_positions(NUM_LINE_POINTS)
        self.line = visuals.Line(line_data, parent=self.view_bot.scene, color=LINE_COLOR_CHOICES[0])
        self.view_bot.camera = "panzoom"
        self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

    def set_image_colormap(self, cmap_name: str):
        print(f"Changing image colormap to {cmap_name}")
        self.image.cmap = cmap_name

    def set_line_color(self, color):
        print(f"Changing line color to {color}")
        self.line.set_data(color=color)


def _generate_random_image_data(shape, dtype=np.float32):
    rng = np.random.default_rng()
    data = rng.random(shape, dtype=dtype)
    return data


def _generate_random_line_positions(num_points, dtype=np.float32):
    rng = np.random.default_rng()
    pos = np.empty((num_points, 2), dtype=np.float32)
    pos[:, 0] = np.arange(num_points)
    pos[:, 1] = rng.random((num_points,), dtype=dtype)
    return pos


if __name__ == "__main__":
    app = use_app("pyqt5")
    app.create()
    win = MyMainWindow()
    win.show()
    app.run()
