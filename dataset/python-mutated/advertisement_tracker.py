"""The bluetooth integration advertisement tracker."""
from __future__ import annotations
from typing import Any
from homeassistant.core import callback
from .models import BluetoothServiceInfoBleak
ADVERTISING_TIMES_NEEDED = 16
TRACKER_BUFFERING_WOBBLE_SECONDS = 5

class AdvertisementTracker:
    """Tracker to determine the interval that a device is advertising."""
    __slots__ = ('intervals', 'fallback_intervals', 'sources', '_timings')

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize the tracker.'
        self.intervals: dict[str, float] = {}
        self.fallback_intervals: dict[str, float] = {}
        self.sources: dict[str, str] = {}
        self._timings: dict[str, list[float]] = {}

    @callback
    def async_diagnostics(self) -> dict[str, dict[str, Any]]:
        if False:
            while True:
                i = 10
        'Return diagnostics.'
        return {'intervals': self.intervals, 'fallback_intervals': self.fallback_intervals, 'sources': self.sources, 'timings': self._timings}

    @callback
    def async_collect(self, service_info: BluetoothServiceInfoBleak) -> None:
        if False:
            print('Hello World!')
        'Collect timings for the tracker.\n\n        For performance reasons, it is the responsibility of the\n        caller to check if the device already has an interval set or\n        the source has changed before calling this function.\n        '
        address = service_info.address
        self.sources[address] = service_info.source
        timings = self._timings.setdefault(address, [])
        timings.append(service_info.time)
        if len(timings) != ADVERTISING_TIMES_NEEDED:
            return
        max_time_between_advertisements = timings[1] - timings[0]
        for i in range(2, len(timings)):
            time_between_advertisements = timings[i] - timings[i - 1]
            if time_between_advertisements > max_time_between_advertisements:
                max_time_between_advertisements = time_between_advertisements
        self.intervals[address] = max_time_between_advertisements
        del self._timings[address]

    @callback
    def async_remove_address(self, address: str) -> None:
        if False:
            print('Hello World!')
        'Remove the tracker.'
        self.intervals.pop(address, None)
        self.sources.pop(address, None)
        self._timings.pop(address, None)

    @callback
    def async_remove_fallback_interval(self, address: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove fallback interval.'
        self.fallback_intervals.pop(address, None)

    @callback
    def async_remove_source(self, source: str) -> None:
        if False:
            while True:
                i = 10
        'Remove the tracker.'
        for (address, tracked_source) in list(self.sources.items()):
            if tracked_source == source:
                self.async_remove_address(address)