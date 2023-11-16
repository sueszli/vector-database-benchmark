from urh.signalprocessing.IQArray import IQArray
from urh.ui.painting.SceneManager import SceneManager

class SniffSceneManager(SceneManager):

    def __init__(self, data_array, parent, window_length=5 * 10 ** 6):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.data_array = data_array
        self.__start = 0
        self.__end = 0
        self.window_length = window_length

    @property
    def plot_data(self):
        if False:
            i = 10
            return i + 15
        return self.data_array[self.__start:self.end]

    @plot_data.setter
    def plot_data(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def end(self):
        if False:
            print('Hello World!')
        return self.__end

    @end.setter
    def end(self, value):
        if False:
            return 10
        if value > self.window_length:
            self.__start = value - self.window_length
        else:
            self.__start = 0
        self.__end = value