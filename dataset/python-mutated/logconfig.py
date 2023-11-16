""" Configure the logging system for Bokeh.

By default, logging is not configured, to allow users of Bokeh to have full
control over logging policy. However, it is useful to be able to enable
logging arbitrarily during when developing Bokeh. This can be accomplished
by setting the environment variable ``BOKEH_PY_LOG_LEVEL``. Valid values are,
in order of increasing severity:

- ``debug``
- ``info``
- ``warn``
- ``error``
- ``fatal``
- ``none``

The default logging level is ``none``.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import sys
from typing import Any, cast
from ..settings import settings
__all__ = ('basicConfig',)
default_handler: logging.Handler | None = None

def basicConfig(**kwargs: Any) -> None:
    if False:
        return 10
    '\n    A logging.basicConfig() wrapper that also undoes the default\n    Bokeh-specific configuration.\n    '
    if default_handler is not None:
        bokeh_logger.removeHandler(default_handler)
        bokeh_logger.propagate = True
    logging.basicConfig(**kwargs)
TRACE = 9
logging.addLevelName(TRACE, 'TRACE')

def trace(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
    if False:
        i = 10
        return i + 15
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kws)
cast(Any, logging).Logger.trace = trace
cast(Any, logging).TRACE = TRACE
level = settings.py_log_level()
bokeh_logger = logging.getLogger('bokeh')
root_logger = logging.getLogger()
if level is not None:
    bokeh_logger.setLevel(level)
if not (root_logger.handlers or bokeh_logger.handlers):
    default_handler = logging.StreamHandler(sys.stderr)
    default_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    bokeh_logger.addHandler(default_handler)
    bokeh_logger.propagate = False