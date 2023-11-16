import logging
from typing import Any
from typing import cast
from typing import Optional

class VerboseLogging(logging.Logger):
    """
    Extend logging to add a verbose logging level that is between
    INFO and DEBUG.

    Also expose a logging.verbose() method so there is no need
    to call log(VERBOSE_LEVEL, msg) every time
    """
    VERBOSE_LOG_LEVEL = 15

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def verbose(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.isEnabledFor(self.VERBOSE_LOG_LEVEL):
            self._log(self.VERBOSE_LOG_LEVEL, msg, args, **kwargs)

def install_verbose_logging() -> None:
    if False:
        i = 10
        return i + 15
    "\n    Makes 3 changes to stdlib logging:\n    - add in logging.VERBOSE constant\n    - add VERBOSE as a logging level\n    - set VerboseLogging as default class returned by logging.getLogger\n        - thus exposing logger.verbose(msg) method\n\n    Any calls to getLogger before this method returns will return base\n    logging.Logger class that doesn't have verbose() convenience method\n    "
    logging.VERBOSE = 15
    logging.addLevelName(VerboseLogging.VERBOSE_LOG_LEVEL, 'VERBOSE')
    logging.setLoggerClass(VerboseLogging)
install_verbose_logging()

def getLogger(name: Optional[str]) -> VerboseLogging:
    if False:
        while True:
            i = 10
    '\n    Wrapper around logging.getLogger to correctly cast so mypy\n    detects verbose() function\n    '
    return cast(VerboseLogging, logging.getLogger(name))