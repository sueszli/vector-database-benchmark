"""A collection of WatermarkEstimator implementations that SplittableDoFns
can use."""
from apache_beam.io.iobase import WatermarkEstimator
from apache_beam.transforms.core import WatermarkEstimatorProvider
from apache_beam.utils.timestamp import Timestamp

class MonotonicWatermarkEstimator(WatermarkEstimator):
    """A WatermarkEstimator which assumes that timestamps of all ouput records
  are increasing monotonically.
  """

    def __init__(self, timestamp):
        if False:
            i = 10
            return i + 15
        'For a new <element, restriction> pair, the initial value is None. When\n    resuming processing, the initial timestamp will be the last reported\n    watermark.\n    '
        self._watermark = timestamp
        self._last_observed_timestamp = timestamp

    def observe_timestamp(self, timestamp):
        if False:
            while True:
                i = 10
        self._last_observed_timestamp = timestamp

    def current_watermark(self):
        if False:
            i = 10
            return i + 15
        if self._last_observed_timestamp is not None and self._last_observed_timestamp >= self._watermark:
            self._watermark = self._last_observed_timestamp
        return self._watermark

    def get_estimator_state(self):
        if False:
            for i in range(10):
                print('nop')
        return self._watermark

    @staticmethod
    def default_provider():
        if False:
            i = 10
            return i + 15
        'Provide a default WatermarkEstimatorProvider for\n    MonotonicWatermarkEstimator.\n    '

        class DefaultMonotonicWatermarkEstimator(WatermarkEstimatorProvider):

            def initial_estimator_state(self, element, restriction):
                if False:
                    i = 10
                    return i + 15
                return None

            def create_watermark_estimator(self, estimator_state):
                if False:
                    i = 10
                    return i + 15
                return MonotonicWatermarkEstimator(estimator_state)
        return DefaultMonotonicWatermarkEstimator()

class WalltimeWatermarkEstimator(WatermarkEstimator):
    """A WatermarkEstimator which uses processing time as the estimated watermark.
  """

    def __init__(self, timestamp=None):
        if False:
            return 10
        self._timestamp = timestamp or Timestamp.now()

    def observe_timestamp(self, timestamp):
        if False:
            print('Hello World!')
        pass

    def current_watermark(self):
        if False:
            for i in range(10):
                print('nop')
        self._timestamp = max(self._timestamp, Timestamp.now())
        return self._timestamp

    def get_estimator_state(self):
        if False:
            print('Hello World!')
        return self._timestamp

    @staticmethod
    def default_provider():
        if False:
            print('Hello World!')
        'Provide a default WatermarkEstimatorProvider for\n    WalltimeWatermarkEstimator.\n    '

        class DefaultWalltimeWatermarkEstimator(WatermarkEstimatorProvider):

            def initial_estimator_state(self, element, restriction):
                if False:
                    i = 10
                    return i + 15
                return None

            def create_watermark_estimator(self, estimator_state):
                if False:
                    while True:
                        i = 10
                return WalltimeWatermarkEstimator(estimator_state)
        return DefaultWalltimeWatermarkEstimator()

class ManualWatermarkEstimator(WatermarkEstimator):
    """A WatermarkEstimator which is controlled manually from within a DoFn.

  The DoFn must invoke set_watermark to advance the watermark.
  """

    def __init__(self, watermark):
        if False:
            for i in range(10):
                print('nop')
        self._watermark = watermark

    def observe_timestamp(self, timestamp):
        if False:
            return 10
        pass

    def current_watermark(self):
        if False:
            while True:
                i = 10
        return self._watermark

    def get_estimator_state(self):
        if False:
            return 10
        return self._watermark

    def set_watermark(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        'Sets a timestamp before or at the timestamps of all future elements\n    produced by the associated DoFn.\n\n    This can be approximate. If records are output that violate this guarantee,\n    they will be considered late, which will affect how they will be processed.\n    See https://beam.apache.org/documentation/programming-guide/#watermarks-and-late-data\n    for more information on late data and how to handle it.\n\n    However, this value should be as late as possible. Downstream windows may\n    not be able to close until this watermark passes their end.\n    '
        if not isinstance(timestamp, Timestamp):
            raise ValueError('set_watermark expects a Timestamp as input')
        if self._watermark and self._watermark > timestamp:
            raise ValueError('Watermark must be monotonically increasing.Provided watermark %s is less than current watermark %s', timestamp, self._watermark)
        self._watermark = timestamp

    @staticmethod
    def default_provider():
        if False:
            i = 10
            return i + 15
        'Provide a default WatermarkEstimatorProvider for\n    WalltimeWatermarkEstimator.\n    '

        class DefaultManualWatermarkEstimatorProvider(WatermarkEstimatorProvider):

            def initial_estimator_state(self, element, restriction):
                if False:
                    i = 10
                    return i + 15
                return None

            def create_watermark_estimator(self, estimator_state):
                if False:
                    i = 10
                    return i + 15
                return ManualWatermarkEstimator(estimator_state)
        return DefaultManualWatermarkEstimatorProvider()