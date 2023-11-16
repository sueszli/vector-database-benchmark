from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from . import TextLogLevel

class TextLogLevelExt:
    """Extension for [TextLogLevel][rerun.components.TextLogLevel]."""
    CRITICAL: TextLogLevel = None
    ' Designates catastrophic failures. '
    ERROR: TextLogLevel = None
    ' Designates very serious errors. '
    WARN: TextLogLevel = None
    ' Designates hazardous situations. '
    INFO: TextLogLevel = None
    ' Designates useful information. '
    DEBUG: TextLogLevel = None
    ' Designates lower priority information. '
    TRACE: TextLogLevel = None
    ' Designates very low priority, often extremely verbose, information. '

    @staticmethod
    def deferred_patch_class(cls: Any) -> None:
        if False:
            i = 10
            return i + 15
        cls.CRITICAL = cls('CRITICAL')
        cls.ERROR = cls('ERROR')
        cls.WARN = cls('WARN')
        cls.INFO = cls('INFO')
        cls.DEBUG = cls('DEBUG')
        cls.TRACE = cls('TRACE')