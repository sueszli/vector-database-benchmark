"""Module to expose options that control how Interactive Beam works.

For internal use only; no backwards-compatibility guarantees.
"""
from dateutil import tz
from apache_beam.runners.interactive.options import capture_control

class InteractiveOptions(object):
    """An intermediate facade to query and configure options that guide how
  Interactive Beam works."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._capture_control = capture_control.CaptureControl()
        self._display_timestamp_format = '%Y-%m-%d %H:%M:%S.%f%z'
        self._display_timezone = tz.tzlocal()
        self._cache_root = None

    def __repr__(self):
        if False:
            return 10
        options_str = '\n'.join(('{} = {}'.format(k, getattr(self, k)) for k in dir(self) if k[0] != '_' and k != 'capture_control'))
        return 'interactive_beam.options:\n{}'.format(options_str)

    @property
    def capture_control(self):
        if False:
            i = 10
            return i + 15
        return self._capture_control