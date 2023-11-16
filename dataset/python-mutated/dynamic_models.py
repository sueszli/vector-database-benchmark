from abc import ABCMeta, abstractmethod
import torch
from torch import nn
from torch.nn import Parameter
import pyro.distributions as dist
from pyro.distributions.util import eye_like

class DynamicModel(nn.Module, metaclass=ABCMeta):
    """
    Dynamic model interface.

    :param dimension: native state dimension.
    :param dimension_pv: PV state dimension.
    :param num_process_noise_parameters: process noise parameter space dimension.
          This for UKF applications. Can be left as ``None`` for EKF and most
          other filters.
    """

    def __init__(self, dimension, dimension_pv, num_process_noise_parameters=None):
        if False:
            while True:
                i = 10
        self._dimension = dimension
        self._dimension_pv = dimension_pv
        self._num_process_noise_parameters = num_process_noise_parameters
        super().__init__()

    @property
    def dimension(self):
        if False:
            i = 10
            return i + 15
        '\n        Native state dimension access.\n        '
        return self._dimension

    @property
    def dimension_pv(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        PV state dimension access.\n        '
        return self._dimension_pv

    @property
    def num_process_noise_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process noise parameters space dimension access.\n        '
        return self._num_process_noise_parameters

    @abstractmethod
    def forward(self, x, dt, do_normalization=True):
        if False:
            print('Hello World!')
        "\n        Integrate native state ``x`` over time interval ``dt``.\n\n        :param x: current native state. If the DynamicModel is non-differentiable,\n              be sure to handle the case of ``x`` being augmented with process\n              noise parameters.\n        :param dt: time interval to integrate over.\n        :param do_normalization: whether to perform normalization on output, e.g.,\n              mod'ing angles into an interval.\n        :return: Native state x integrated dt into the future.\n        "
        raise NotImplementedError

    def geodesic_difference(self, x1, x0):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return the geodesic difference between 2 native states.\n        This is a generalization of the Euclidean operation ``x1 - x0``.\n\n        :param x1: native state.\n        :param x0: native state.\n        :return: Geodesic difference between native states ``x1`` and ``x2``.\n        '
        return x1 - x0

    @abstractmethod
    def mean2pv(self, x):
        if False:
            while True:
                i = 10
        '\n        Compute and return PV state from native state. Useful for combining\n        state estimates of different types in IMM (Interacting Multiple Model)\n        filtering.\n\n        :param x: native state estimate mean.\n        :return: PV state estimate mean.\n        '
        raise NotImplementedError

    @abstractmethod
    def cov2pv(self, P):
        if False:
            while True:
                i = 10
        '\n        Compute and return PV covariance from native covariance. Useful for\n        combining state estimates of different types in IMM (Interacting\n        Multiple Model) filtering.\n\n        :param P: native state estimate covariance.\n        :return: PV state estimate covariance.\n        '
        raise NotImplementedError

    @abstractmethod
    def process_noise_cov(self, dt=0.0):
        if False:
            return 10
        '\n        Compute and return process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q). For a DifferentiableDynamicModel, this is\n            the covariance of the native state ``x`` resulting from stochastic\n            integration (for use with EKF). Otherwise, it is the covariance\n            directly of the process noise parameters (for use with UKF).\n        '
        raise NotImplementedError

    def process_noise_dist(self, dt=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Return a distribution object of state displacement from the process noise\n        distribution over a time interval.\n\n        :param dt: time interval that process noise accumulates over.\n        :return: :class:`~pyro.distributions.torch.MultivariateNormal`.\n        '
        Q = self.process_noise_cov(dt)
        return dist.MultivariateNormal(torch.zeros(Q.shape[-1], dtype=Q.dtype, device=Q.device), Q)

class DifferentiableDynamicModel(DynamicModel):
    """
    DynamicModel for which state transition Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    """

    @abstractmethod
    def jacobian(self, dt):
        if False:
            print('Hello World!')
        '\n        Compute and return native state transition Jacobian (F) over time\n        interval ``dt``.\n\n        :param  dt: time interval to integrate over.\n        :return: Read-only Jacobian (F) of integration map (f).\n        '
        raise NotImplementedError

class Ncp(DifferentiableDynamicModel):
    """
    NCP (Nearly-Constant Position) dynamic model. May be subclassed, e.g., with
    CWNV (Continuous White Noise Velocity) or DWNV (Discrete White Noise
    Velocity).

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    """

    def __init__(self, dimension, sv2):
        if False:
            while True:
                i = 10
        dimension_pv = 2 * dimension
        super().__init__(dimension, dimension_pv, num_process_noise_parameters=1)
        if not isinstance(sv2, torch.Tensor):
            sv2 = torch.tensor(sv2)
        self.sv2 = Parameter(sv2)
        self._F_cache = eye_like(sv2, dimension)
        self._Q_cache = {}

    def forward(self, x, dt, do_normalization=True):
        if False:
            return 10
        "\n        Integrate native state ``x`` over time interval ``dt``.\n\n        :param x: current native state. If the DynamicModel is non-differentiable,\n              be sure to handle the case of ``x`` being augmented with process\n              noise parameters.\n        :param dt: time interval to integrate over.\n            do_normalization: whether to perform normalization on output, e.g.,\n            mod'ing angles into an interval. Has no effect for this subclass.\n        :return: Native state x integrated dt into the future.\n        "
        return x

    def mean2pv(self, x):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return PV state from native state. Useful for combining\n        state estimates of different types in IMM (Interacting Multiple Model)\n        filtering.\n\n        :param x: native state estimate mean.\n        :return: PV state estimate mean.\n        '
        with torch.no_grad():
            x_pv = torch.zeros(2 * self._dimension, dtype=x.dtype, device=x.device)
            x_pv[:self._dimension] = x
        return x_pv

    def cov2pv(self, P):
        if False:
            while True:
                i = 10
        '\n        Compute and return PV covariance from native covariance. Useful for\n        combining state estimates of different types in IMM (Interacting\n        Multiple Model) filtering.\n\n        :param P: native state estimate covariance.\n        :return: PV state estimate covariance.\n        '
        d = 2 * self._dimension
        with torch.no_grad():
            P_pv = torch.zeros(d, d, dtype=P.dtype, device=P.device)
            P_pv[:self._dimension, :self._dimension] = P
        return P_pv

    def jacobian(self, dt):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return cached native state transition Jacobian (F) over\n        time interval ``dt``.\n\n        :param dt: time interval to integrate over.\n        :return: Read-only Jacobian (F) of integration map (f).\n        '
        return self._F_cache

    @abstractmethod
    def process_noise_cov(self, dt=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q) of the native state ``x`` resulting from\n            stochastic integration (for use with EKF).\n        '
        raise NotImplementedError

class Ncv(DifferentiableDynamicModel):
    """
    NCV (Nearly-Constant Velocity) dynamic model. May be subclassed, e.g., with
    CWNA (Continuous White Noise Acceleration) or DWNA (Discrete White Noise
    Acceleration).

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    """

    def __init__(self, dimension, sa2):
        if False:
            while True:
                i = 10
        dimension_pv = dimension
        super().__init__(dimension, dimension_pv, num_process_noise_parameters=1)
        if not isinstance(sa2, torch.Tensor):
            sa2 = torch.tensor(sa2)
        self.sa2 = Parameter(sa2)
        self._F_cache = {}
        self._Q_cache = {}

    def forward(self, x, dt, do_normalization=True):
        if False:
            i = 10
            return i + 15
        "\n        Integrate native state ``x`` over time interval ``dt``.\n\n        :param x: current native state. If the DynamicModel is non-differentiable,\n              be sure to handle the case of ``x`` being augmented with process\n              noise parameters.\n        :param dt: time interval to integrate over.\n        :param do_normalization: whether to perform normalization on output, e.g.,\n              mod'ing angles into an interval. Has no effect for this subclass.\n\n        :return: Native state x integrated dt into the future.\n        "
        F = self.jacobian(dt)
        return F.mm(x.unsqueeze(1)).squeeze(1)

    def mean2pv(self, x):
        if False:
            while True:
                i = 10
        '\n        Compute and return PV state from native state. Useful for combining\n        state estimates of different types in IMM (Interacting Multiple Model)\n        filtering.\n\n        :param x: native state estimate mean.\n        :return: PV state estimate mean.\n        '
        return x

    def cov2pv(self, P):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return PV covariance from native covariance. Useful for\n        combining state estimates of different types in IMM (Interacting\n        Multiple Model) filtering.\n\n        :param P: native state estimate covariance.\n        :return: PV state estimate covariance.\n        '
        return P

    def jacobian(self, dt):
        if False:
            print('Hello World!')
        '\n        Compute and return cached native state transition Jacobian (F) over\n        time interval ``dt``.\n\n        :param dt: time interval to integrate over.\n        :return: Read-only Jacobian (F) of integration map (f).\n        '
        if dt not in self._F_cache:
            d = self._dimension
            with torch.no_grad():
                F = eye_like(self.sa2, d)
                F[:d // 2, d // 2:] = dt * eye_like(self.sa2, d // 2)
            self._F_cache[dt] = F
        return self._F_cache[dt]

    @abstractmethod
    def process_noise_cov(self, dt=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q) of the native state ``x`` resulting from\n            stochastic integration (for use with EKF).\n        '
        raise NotImplementedError

class NcpContinuous(Ncp):
    """
    NCP (Nearly-Constant Position) dynamic model with CWNV (Continuous White
    Noise Velocity).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    """

    def process_noise_cov(self, dt=0.0):
        if False:
            print('Hello World!')
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q) of the native state ``x`` resulting from\n            stochastic integration (for use with EKF).\n        '
        if dt not in self._Q_cache:
            q = self.sv2 * dt
            Q = q * dt * eye_like(self.sv2, self._dimension)
            self._Q_cache[dt] = Q
        return self._Q_cache[dt]

class NcvContinuous(Ncv):
    """
    NCV (Nearly-Constant Velocity) dynamic model with CWNA (Continuous White
    Noise Acceleration).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    """

    def process_noise_cov(self, dt=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n\n        :return: Read-only covariance (Q) of the native state ``x`` resulting from\n            stochastic integration (for use with EKF).\n        '
        if dt not in self._Q_cache:
            with torch.no_grad():
                d = self._dimension
                dt2 = dt * dt
                dt3 = dt2 * dt
                Q = torch.zeros(d, d, dtype=self.sa2.dtype, device=self.sa2.device)
                eye = eye_like(self.sa2, d // 2)
                Q[:d // 2, :d // 2] = dt3 * eye / 3.0
                Q[:d // 2, d // 2:] = dt2 * eye / 2.0
                Q[d // 2:, :d // 2] = dt2 * eye / 2.0
                Q[d // 2:, d // 2:] = dt * eye
            Q = Q * (self.sa2 * dt)
            self._Q_cache[dt] = Q
        return self._Q_cache[dt]

class NcpDiscrete(Ncp):
    """
    NCP (Nearly-Constant Position) dynamic model with DWNV (Discrete White
    Noise Velocity).

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.
    """

    def process_noise_cov(self, dt=0.0):
        if False:
            return 10
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q) of the native state `x` resulting from\n            stochastic integration (for use with EKF).\n        '
        if dt not in self._Q_cache:
            Q = self.sv2 * dt * dt * eye_like(self.sv2, self._dimension)
            self._Q_cache[dt] = Q
        return self._Q_cache[dt]

class NcvDiscrete(Ncv):
    """
    NCV (Nearly-Constant Velocity) dynamic model with DWNA (Discrete White
    Noise Acceleration).

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.
    """

    def process_noise_cov(self, dt=0.0):
        if False:
            print('Hello World!')
        '\n        Compute and return cached process noise covariance (Q).\n\n        :param dt: time interval to integrate over.\n        :return: Read-only covariance (Q) of the native state `x` resulting from\n            stochastic integration (for use with EKF). (Note that this Q, modulo\n            numerical error, has rank `dimension/2`. So, it is only positive\n            semi-definite.)\n        '
        if dt not in self._Q_cache:
            with torch.no_grad():
                d = self._dimension
                dt2 = dt * dt
                dt3 = dt2 * dt
                dt4 = dt2 * dt2
                Q = torch.zeros(d, d, dtype=self.sa2.dtype, device=self.sa2.device)
                Q[:d // 2, :d // 2] = 0.25 * dt4 * eye_like(self.sa2, d // 2)
                Q[:d // 2, d // 2:] = 0.5 * dt3 * eye_like(self.sa2, d // 2)
                Q[d // 2:, :d // 2] = 0.5 * dt3 * eye_like(self.sa2, d // 2)
                Q[d // 2:, d // 2:] = dt2 * eye_like(self.sa2, d // 2)
            Q = Q * self.sa2
            self._Q_cache[dt] = Q
        return self._Q_cache[dt]