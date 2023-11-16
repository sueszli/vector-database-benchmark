from urh.signalprocessing.IQArray import IQArray
from urh.ui.painting.SceneManager import SceneManager
from urh.util.RingBuffer import RingBuffer

class ContinuousSceneManager(SceneManager):

    def __init__(self, ring_buffer: RingBuffer, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ring_buffer = ring_buffer
        self.__start = 0
        self.__end = 0

    @property
    def plot_data(self):
        if False:
            print('Hello World!')
        return self.ring_buffer.view_data.real

    @plot_data.setter
    def plot_data(self, value):
        if False:
            i = 10
            return i + 15
        pass

    @property
    def end(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ring_buffer.size

    @end.setter
    def end(self, value):
        if False:
            i = 10
            return i + 15
        pass