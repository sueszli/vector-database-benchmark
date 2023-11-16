from abc import ABCMeta, abstractmethod
from bigdl.orca.tfpark import KerasModel as TFParkKerasModel
import tensorflow as tf
from bigdl.chronos.forecaster.abstract import Forecaster

class TFParkForecaster(TFParkKerasModel, Forecaster, metaclass=ABCMeta):
    """
    Base class for TFPark KerasModel based Forecast models.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a tf.keras model.\n        Turns the tf.keras model returned from _build into a tfpark.KerasModel\n        '
        self.model = self._build()
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError((isinstance(self.model, tf.keras.Model), 'expect model is tf.keras.Model'))
        super().__init__(self.model)

    @abstractmethod
    def _build(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build a tf.keras model.\n\n        :return: a tf.keras model (compiled)\n        '
        pass