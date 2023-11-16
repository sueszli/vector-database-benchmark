"""Common utility class to help SDK harness to execute an SDF. """
import logging
import threading
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union
from apache_beam.transforms.core import WatermarkEstimatorProvider
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp
from apache_beam.utils.windowed_value import WindowedValue
if TYPE_CHECKING:
    from apache_beam.io.iobase import RestrictionProgress
    from apache_beam.io.iobase import RestrictionTracker
    from apache_beam.io.iobase import WatermarkEstimator
_LOGGER = logging.getLogger(__name__)
SplitResultPrimary = NamedTuple('SplitResultPrimary', [('primary_value', WindowedValue)])
SplitResultResidual = NamedTuple('SplitResultResidual', [('residual_value', WindowedValue), ('current_watermark', Timestamp), ('deferred_timestamp', Optional[Duration])])

class ThreadsafeRestrictionTracker(object):
    """A thread-safe wrapper which wraps a `RestrictionTracker`.

  This wrapper guarantees synchronization of modifying restrictions across
  multi-thread.
  """

    def __init__(self, restriction_tracker):
        if False:
            for i in range(10):
                print('nop')
        from apache_beam.io.iobase import RestrictionTracker
        if not isinstance(restriction_tracker, RestrictionTracker):
            raise ValueError('Initialize ThreadsafeRestrictionTracker requiresRestrictionTracker.')
        self._restriction_tracker = restriction_tracker
        self._timestamp = None
        self._lock = threading.RLock()
        self._deferred_residual = None
        self._deferred_timestamp = None

    def current_restriction(self):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            return self._restriction_tracker.current_restriction()

    def try_claim(self, position):
        if False:
            while True:
                i = 10
        with self._lock:
            return self._restriction_tracker.try_claim(position)

    def defer_remainder(self, deferred_time=None):
        if False:
            i = 10
            return i + 15
        'Performs self-checkpoint on current processing restriction with an\n    expected resuming time.\n\n    Self-checkpoint could happen during processing elements. When executing an\n    DoFn.process(), you may want to stop processing an element and resuming\n    later if current element has been processed quit a long time or you also\n    want to have some outputs from other elements. ``defer_remainder()`` can be\n    called on per element if needed.\n\n    Args:\n      deferred_time: A relative ``Duration`` that indicates the ideal time gap\n        between now and resuming, or an absolute ``Timestamp`` for resuming\n        execution time. If the time_delay is None, the deferred work will be\n        executed as soon as possible.\n    '
        with self._lock:
            self._timestamp = Timestamp.now()
            if deferred_time and (not isinstance(deferred_time, (Duration, Timestamp))):
                raise ValueError('The timestamp of deter_remainder() should be a Duration or a Timestamp, or None.')
            self._deferred_timestamp = deferred_time
            checkpoint = self.try_split(0)
            if checkpoint:
                (_, self._deferred_residual) = checkpoint

    def check_done(self):
        if False:
            i = 10
            return i + 15
        with self._lock:
            return self._restriction_tracker.check_done()

    def current_progress(self):
        if False:
            i = 10
            return i + 15
        with self._lock:
            return self._restriction_tracker.current_progress()

    def try_split(self, fraction_of_remainder):
        if False:
            return 10
        with self._lock:
            return self._restriction_tracker.try_split(fraction_of_remainder)

    def deferred_status(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns deferred work which is produced by ``defer_remainder()``.\n\n    When there is a self-checkpoint performed, the system needs to fulfill the\n    DelayedBundleApplication with deferred_work for a  ProcessBundleResponse.\n    The system calls this API to get deferred_residual with watermark together\n    to help the runner to schedule a future work.\n\n    Returns: (deferred_residual, time_delay) if having any residual, else None.\n    '
        if self._deferred_residual:
            if not self._deferred_timestamp:
                self._deferred_timestamp = Duration()
            elif isinstance(self._deferred_timestamp, Timestamp):
                self._deferred_timestamp = self._deferred_timestamp - Timestamp.now()
            elif isinstance(self._deferred_timestamp, Duration):
                self._deferred_timestamp -= Timestamp.now() - self._timestamp
            return (self._deferred_residual, self._deferred_timestamp)
        return None

    def is_bounded(self):
        if False:
            i = 10
            return i + 15
        return self._restriction_tracker.is_bounded()

class RestrictionTrackerView(object):
    """A DoFn view of thread-safe RestrictionTracker.

  The RestrictionTrackerView wraps a ThreadsafeRestrictionTracker and only
  exposes APIs that will be called by a ``DoFn.process()``. During execution
  time, the RestrictionTrackerView will be fed into the ``DoFn.process`` as a
  restriction_tracker.
  """

    def __init__(self, threadsafe_restriction_tracker):
        if False:
            i = 10
            return i + 15
        if not isinstance(threadsafe_restriction_tracker, ThreadsafeRestrictionTracker):
            raise ValueError('Initialize RestrictionTrackerView requires ThreadsafeRestrictionTracker.')
        self._threadsafe_restriction_tracker = threadsafe_restriction_tracker

    def current_restriction(self):
        if False:
            i = 10
            return i + 15
        return self._threadsafe_restriction_tracker.current_restriction()

    def try_claim(self, position):
        if False:
            print('Hello World!')
        return self._threadsafe_restriction_tracker.try_claim(position)

    def defer_remainder(self, deferred_time=None):
        if False:
            for i in range(10):
                print('nop')
        self._threadsafe_restriction_tracker.defer_remainder(deferred_time)

    def is_bounded(self):
        if False:
            while True:
                i = 10
        self._threadsafe_restriction_tracker.is_bounded()

class ThreadsafeWatermarkEstimator(object):
    """A threadsafe wrapper which wraps a WatermarkEstimator with locking
  mechanism to guarantee multi-thread safety.
  """

    def __init__(self, watermark_estimator):
        if False:
            return 10
        from apache_beam.io.iobase import WatermarkEstimator
        if not isinstance(watermark_estimator, WatermarkEstimator):
            raise ValueError('Initializing Threadsafe requires a WatermarkEstimator')
        self._watermark_estimator = watermark_estimator
        self._lock = threading.Lock()

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        if hasattr(self._watermark_estimator, attr):

            def method_wrapper(*args, **kw):
                if False:
                    print('Hello World!')
                with self._lock:
                    return getattr(self._watermark_estimator, attr)(*args, **kw)
            return method_wrapper
        raise AttributeError(attr)

    def get_estimator_state(self):
        if False:
            print('Hello World!')
        with self._lock:
            return self._watermark_estimator.get_estimator_state()

    def current_watermark(self):
        if False:
            return 10
        with self._lock:
            return self._watermark_estimator.current_watermark()

    def observe_timestamp(self, timestamp):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(timestamp, Timestamp):
            raise ValueError('Input of observe_timestamp should be a Timestamp object')
        with self._lock:
            self._watermark_estimator.observe_timestamp(timestamp)

class NoOpWatermarkEstimatorProvider(WatermarkEstimatorProvider):
    """A WatermarkEstimatorProvider which creates NoOpWatermarkEstimator for the
  framework.
  """

    def initial_estimator_state(self, element, restriction):
        if False:
            print('Hello World!')
        return None

    def create_watermark_estimator(self, estimator_state):
        if False:
            print('Hello World!')
        from apache_beam.io.iobase import WatermarkEstimator

        class _NoOpWatermarkEstimator(WatermarkEstimator):
            """A No-op WatermarkEstimator which is provided for the framework if there
      is no custom one.
      """

            def observe_timestamp(self, timestamp):
                if False:
                    while True:
                        i = 10
                pass

            def current_watermark(self):
                if False:
                    i = 10
                    return i + 15
                return None

            def get_estimator_state(self):
                if False:
                    return 10
                return None
        return _NoOpWatermarkEstimator()