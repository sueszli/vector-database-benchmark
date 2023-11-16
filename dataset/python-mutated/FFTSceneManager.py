import numpy as np
from PyQt5.QtGui import QPainterPath, QPen
from PyQt5.QtWidgets import QGraphicsPathItem
from urh import settings
from urh.cythonext import path_creator
from urh.ui.painting.GridScene import GridScene
from urh.ui.painting.SceneManager import SceneManager

class FFTSceneManager(SceneManager):

    def __init__(self, parent, graphic_view=None):
        if False:
            print('Hello World!')
        self.peak = []
        super().__init__(parent)
        self.scene = GridScene(parent=graphic_view)
        self.scene.setBackgroundBrush(settings.BGCOLOR)
        self.peak_item = self.scene.addPath(QPainterPath(), QPen(settings.PEAK_COLOR, 0))

    def show_scene_section(self, x1: float, x2: float, subpath_ranges=None, colors=None):
        if False:
            return 10
        start = int(x1) if x1 > 0 else 0
        end = int(x2) if x2 < self.num_samples else self.num_samples
        paths = path_creator.create_path(np.log10(self.plot_data), start, end)
        self.set_path(paths, colors=None)
        try:
            if len(self.peak) > 0:
                peak_path = path_creator.create_path(np.log10(self.peak), start, end)[0]
                self.peak_item.setPath(peak_path)
        except RuntimeWarning:
            pass

    def init_scene(self, draw_grid=True):
        if False:
            for i in range(10):
                print('nop')
        self.scene.draw_grid = draw_grid
        self.peak = self.plot_data if len(self.peak) < self.num_samples else np.maximum(self.peak, self.plot_data)
        self.scene.setSceneRect(0, -5, self.num_samples, 10)

    def clear_path(self):
        if False:
            print('Hello World!')
        for item in self.scene.items():
            if isinstance(item, QGraphicsPathItem) and item != self.peak_item:
                self.scene.removeItem(item)
                item.setParentItem(None)
                del item

    def clear_peak(self):
        if False:
            return 10
        self.peak = []
        if self.peak_item:
            self.peak_item.setPath(QPainterPath())

    def eliminate(self):
        if False:
            i = 10
            return i + 15
        super().eliminate()
        self.peak = None
        self.peak_item = None