import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem, QApplication
from urh.signalprocessing.Spectrogram import Spectrogram
from urh.ui.painting.SceneManager import SceneManager
from urh.ui.painting.SpectrogramScene import SpectrogramScene

class SpectrogramSceneManager(SceneManager):

    def __init__(self, samples, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.samples_need_update = True
        self.scene.clear()
        self.spectrogram = Spectrogram(samples)
        self.scene = SpectrogramScene()

    @property
    def num_samples(self):
        if False:
            print('Hello World!')
        return len(self.spectrogram.samples)

    def set_parameters(self, samples: np.ndarray, window_size, data_min, data_max) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return true if redraw is needed\n        '
        redraw_needed = False
        if self.samples_need_update:
            self.spectrogram.samples = samples
            redraw_needed = True
            self.samples_need_update = False
        if window_size != self.spectrogram.window_size:
            self.spectrogram.window_size = window_size
            redraw_needed = True
        if data_min != self.spectrogram.data_min:
            self.spectrogram.data_min = data_min
            redraw_needed = True
        if data_max != self.spectrogram.data_max:
            self.spectrogram.data_max = data_max
            redraw_needed = True
        return redraw_needed

    def show_scene_section(self, x1: float, x2: float, subpath_ranges=None, colors=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def update_scene_rect(self):
        if False:
            while True:
                i = 10
        self.scene.setSceneRect(0, 0, self.spectrogram.time_bins, self.spectrogram.freq_bins)

    def show_full_scene(self):
        if False:
            print('Hello World!')
        for item in self.scene.items():
            if isinstance(item, QGraphicsPixmapItem):
                self.scene.removeItem(item)
        x_pos = 0
        for image in self.spectrogram.create_image_segments():
            item = self.scene.addPixmap(QPixmap.fromImage(image))
            item.setPos(x_pos, 0)
            x_pos += image.width()
            QApplication.instance().processEvents()
        self.scene.setSceneRect(0, 0, x_pos, self.spectrogram.freq_bins)

    def init_scene(self):
        if False:
            while True:
                i = 10
        pass

    def eliminate(self):
        if False:
            return 10
        self.spectrogram.samples = None
        self.spectrogram = None
        super().eliminate()