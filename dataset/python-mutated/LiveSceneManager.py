from urh.signalprocessing.IQArray import IQArray
from urh.ui.painting.SceneManager import SceneManager

class LiveSceneManager(SceneManager):

    def __init__(self, data_array, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.plot_data = data_array
        self.end = 0

    @property
    def num_samples(self):
        if False:
            return 10
        return self.end