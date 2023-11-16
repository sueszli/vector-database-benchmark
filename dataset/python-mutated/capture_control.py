"""Module to control how Interactive Beam captures data from sources for
deterministic replayable PCollection evaluation and pipeline runs.

For internal use only; no backwards-compatibility guarantees.
"""
import logging
from datetime import timedelta
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.options import capture_limiters
_LOGGER = logging.getLogger(__name__)

class CaptureControl(object):
    """Options and their utilities that controls how Interactive Beam captures
  deterministic replayable data from sources."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._enable_capture_replay = True
        self._capturable_sources = {ReadFromPubSub}
        self._capture_duration = timedelta(seconds=60)
        self._capture_size_limit = 1000000000.0
        self._test_limiters = None

    def limiters(self):
        if False:
            print('Hello World!')
        if self._test_limiters:
            return self._test_limiters
        return [capture_limiters.SizeLimiter(self._capture_size_limit), capture_limiters.DurationLimiter(self._capture_duration)]

    def set_limiters_for_test(self, limiters):
        if False:
            print('Hello World!')
        self._test_limiters = limiters

def evict_captured_data(pipeline=None):
    if False:
        return 10
    'Evicts all deterministic replayable data that have been captured by\n  Interactive Beam for the given pipeline. If no pipeline is specified, evicts\n  for all user defined pipelines.\n\n  In future PCollection evaluation/visualization and pipeline\n  runs, Interactive Beam will capture fresh data.'
    if ie.current_env().options.enable_recording_replay:
        _LOGGER.info('You have requested Interactive Beam to evict all recorded data that could be deterministically replayed among multiple pipeline runs.')
    ie.current_env().cleanup(pipeline)