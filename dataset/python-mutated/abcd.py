import abc

class BasicUIMeta(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def click(self, x: int, y: int):
        if False:
            while True:
                i = 10
        pass

    @abc.abstractmethod
    def swipe(self, fx: int, fy: int, tx: int, ty: int, duration: float):
        if False:
            return 10
        ' duration is float type, indicate seconds '

    @abc.abstractmethod
    def window_size(self) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        ' return (width, height) '

    @abc.abstractmethod
    def dump_hierarchy(self) -> str:
        if False:
            while True:
                i = 10
        ' return xml content '

    @abc.abstractmethod
    def screenshot(self):
        if False:
            i = 10
            return i + 15
        ' return PIL.Image.Image '