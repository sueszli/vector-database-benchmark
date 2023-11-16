"""Base classes for drift detection.

The _drift_detected and _warning_detected properties are stored as private attributes
and are exposed through the corresponding properties. This is done for documentation
purposes. The properties are not meant to be modified by the user.

"""
from __future__ import annotations
import abc
from . import base

class _BaseDriftDetector(base.Base):
    """Base drift detector.

    This base class is not exposed.

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._drift_detected = False

    def _reset(self):
        if False:
            i = 10
            return i + 15
        "Reset the detector's state."
        self._drift_detected = False

    @property
    def drift_detected(self):
        if False:
            i = 10
            return i + 15
        'Whether or not a drift is detected following the last update.'
        return self._drift_detected

class _BaseDriftAndWarningDetector(_BaseDriftDetector):
    """Base drift detector.

    This base class is not exposed.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._warning_detected = False

    def _reset(self):
        if False:
            return 10
        super()._reset()
        self._warning_detected = False

    @property
    def warning_detected(self):
        if False:
            i = 10
            return i + 15
        'Whether or not a drift is detected following the last update.'
        return self._warning_detected

class DriftDetector(_BaseDriftDetector):
    """A drift detector."""

    @abc.abstractmethod
    def update(self, x: int | float) -> DriftDetector:
        if False:
            for i in range(10):
                print('nop')
        'Update the detector with a single data point.\n\n        Parameters\n        ----------\n        x\n            Input value.\n\n        Returns\n        -------\n        self\n\n        '

class DriftAndWarningDetector(DriftDetector, _BaseDriftAndWarningDetector):
    """A drift detector that is also capable of issuing warnings."""

class BinaryDriftDetector(_BaseDriftDetector):
    """A drift detector for binary data."""

    @abc.abstractmethod
    def update(self, x: bool) -> BinaryDriftDetector:
        if False:
            for i in range(10):
                print('nop')
        'Update the detector with a single boolean input.\n\n        Parameters\n        ----------\n        x\n            Input boolean.\n\n        Returns\n        -------\n        self\n\n        '

class BinaryDriftAndWarningDetector(BinaryDriftDetector, _BaseDriftAndWarningDetector):
    """A binary drift detector that is also capable of issuing warnings."""