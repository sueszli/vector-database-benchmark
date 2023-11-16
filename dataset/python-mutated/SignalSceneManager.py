import math
from urh.signalprocessing.Signal import Signal
from urh.ui.painting.SceneManager import SceneManager

class SignalSceneManager(SceneManager):

    def __init__(self, signal: Signal, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.signal = signal
        self.scene_type = 0
        self.mod_type = 'ASK'

    def show_scene_section(self, x1: float, x2: float, subpath_ranges=None, colors=None):
        if False:
            return 10
        if self.scene_type == 0:
            self.plot_data = self.signal.real_plot_data
        elif self.scene_type == 3:
            self.plot_data = [self.signal.imag_plot_data, self.signal.real_plot_data]
        else:
            self.plot_data = self.signal.qad
        super().show_scene_section(x1, x2, subpath_ranges=subpath_ranges, colors=colors)

    def init_scene(self):
        if False:
            for i in range(10):
                print('nop')
        if self.scene_type == 0:
            self.plot_data = self.signal.real_plot_data
        elif self.scene_type == 3:
            self.plot_data = [self.signal.imag_plot_data, self.signal.real_plot_data]
        else:
            self.plot_data = self.signal.qad
        super().init_scene()
        if self.scene_type == 1 and (self.mod_type == 'FSK' or self.mod_type == 'PSK'):
            self.scene.setSceneRect(0, -4, self.num_samples, 8)
        self.line_item.setLine(0, 0, 0, 0)
        if self.scene_type == 0 or self.scene_type == 3:
            self.scene.draw_noise_area(self.signal.noise_min_plot, self.signal.noise_max_plot - self.signal.noise_min_plot)
        else:
            self.scene.draw_sep_area(-self.signal.center_thresholds)

    def eliminate(self):
        if False:
            i = 10
            return i + 15
        super().eliminate()
        self.signal = None