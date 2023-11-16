from typing import Iterable, Optional
import numpy as np
from numpy import ndarray
from pyspark.mllib.common import callMLlibFunc
from pyspark.rdd import RDD

class KernelDensity:
    """
    Estimate probability density at required points given an RDD of samples
    from the population.

    Examples
    --------
    >>> kd = KernelDensity()
    >>> sample = sc.parallelize([0.0, 1.0])
    >>> kd.setSample(sample)
    >>> kd.estimate([0.0, 1.0])
    array([ 0.12938758,  0.12938758])
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._bandwidth: float = 1.0
        self._sample: Optional[RDD[float]] = None

    def setBandwidth(self, bandwidth: float) -> None:
        if False:
            return 10
        'Set bandwidth of each sample. Defaults to 1.0'
        self._bandwidth = bandwidth

    def setSample(self, sample: RDD[float]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set sample points from the population. Should be a RDD'
        if not isinstance(sample, RDD):
            raise TypeError('samples should be a RDD, received %s' % type(sample))
        self._sample = sample

    def estimate(self, points: Iterable[float]) -> ndarray:
        if False:
            while True:
                i = 10
        'Estimate the probability density at points'
        points = list(points)
        densities = callMLlibFunc('estimateKernelDensity', self._sample, self._bandwidth, points)
        return np.asarray(densities)