from pyflink.metrics import Distribution

class DistributionImpl(Distribution):

    def __init__(self, inner_distribution):
        if False:
            print('Hello World!')
        self._inner_distribution = inner_distribution

    def update(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Updates the distribution value.\n\n        .. versionadded:: 1.11.0\n        '
        self._inner_distribution.update(value)