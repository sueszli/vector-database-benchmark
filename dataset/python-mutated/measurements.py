from abc import ABCMeta, abstractmethod
import torch
from pyro.distributions.util import eye_like

class Measurement(object, metaclass=ABCMeta):
    """
    Gaussian measurement interface.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: continuous time of measurement. If this is not
          provided, `frame_num` must be.
    :param frame_num: discrete time of measurement. If this is not
          provided, `time` must be.
    """

    def __init__(self, mean, cov, time=None, frame_num=None):
        if False:
            i = 10
            return i + 15
        self._dimension = len(mean)
        self._mean = mean
        self._cov = cov
        if time is None and frame_num is None:
            raise ValueError('Must provide time or frame_num!')
        self._time = time
        self._frame_num = frame_num

    @property
    def dimension(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Measurement space dimension access.\n        '
        return self._dimension

    @property
    def mean(self):
        if False:
            i = 10
            return i + 15
        '\n        Measurement mean (``z`` in most Kalman Filtering literature).\n        '
        return self._mean

    @property
    def cov(self):
        if False:
            print('Hello World!')
        '\n        Noise covariance (``R`` in most Kalman Filtering literature).\n        '
        return self._cov

    @property
    def time(self):
        if False:
            return 10
        '\n        Continuous time of measurement.\n        '
        return self._time

    @property
    def frame_num(self):
        if False:
            i = 10
            return i + 15
        '\n        Discrete time of measurement.\n        '
        return self._frame_num

    @abstractmethod
    def __call__(self, x, do_normalization=True):
        if False:
            return 10
        "\n        Measurement map (h) for predicting a measurement ``z`` from target\n        state ``x``.\n\n        :param x: PV state.\n        :param do_normalization: whether to normalize output, e.g., mod'ing angles\n              into an interval.\n        :return Measurement predicted from state ``x``.\n        "
        raise NotImplementedError

    def geodesic_difference(self, z1, z0):
        if False:
            print('Hello World!')
        '\n        Compute and return the geodesic difference between 2 measurements.\n        This is a generalization of the Euclidean operation ``z1 - z0``.\n\n        :param z1: measurement.\n        :param z0: measurement.\n        :return: Geodesic difference between ``z1`` and ``z2``.\n        '
        return z1 - z0

class DifferentiableMeasurement(Measurement):
    """
    Interface for Gaussian measurement for which Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    """

    @abstractmethod
    def jacobian(self, x=None):
        if False:
            print('Hello World!')
        '\n        Compute and return Jacobian (H) of measurement map (h) at target PV\n        state ``x`` .\n\n        :param x: PV state. Use default argument ``None`` when the Jacobian is not\n              state-dependent.\n        :return: Read-only Jacobian (H) of measurement map (h).\n        '
        raise NotImplementedError

class PositionMeasurement(DifferentiableMeasurement):
    """
    Full-rank Gaussian position measurement in Euclidean space.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: time of measurement.
    """

    def __init__(self, mean, cov, time=None, frame_num=None):
        if False:
            i = 10
            return i + 15
        super().__init__(mean, cov, time=time, frame_num=frame_num)
        self._jacobian = torch.cat([eye_like(mean, self.dimension), torch.zeros(self.dimension, self.dimension, dtype=mean.dtype, device=mean.device)], dim=1)

    def __call__(self, x, do_normalization=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Measurement map (h) for predicting a measurement ``z`` from target\n        state ``x``.\n\n        :param x: PV state.\n        :param do_normalization: whether to normalize output. Has no effect for\n              this subclass.\n        :return: Measurement predicted from state ``x``.\n        '
        return x[:self._dimension]

    def jacobian(self, x=None):
        if False:
            print('Hello World!')
        '\n        Compute and return Jacobian (H) of measurement map (h) at target PV\n        state ``x`` .\n\n        :param x: PV state. The default argument ``None`` may be used in this\n              subclass since the Jacobian is not state-dependent.\n        :return: Read-only Jacobian (H) of measurement map (h).\n        '
        return self._jacobian