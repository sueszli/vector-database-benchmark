"""Abstract observer class."""
import abc
from .base_quanter import BaseQuanter

class BaseObserver(BaseQuanter, metaclass=abc.ABCMeta):
    """
    Built-in observers and customized observers should extend this base observer
    and implement abstract methods.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()

    @abc.abstractmethod
    def cal_thresholds(self):
        if False:
            return 10
        pass