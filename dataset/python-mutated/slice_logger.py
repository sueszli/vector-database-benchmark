import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
from airbyte_cdk.models import AirbyteLogMessage, AirbyteMessage, Level
from airbyte_cdk.models import Type as MessageType

class SliceLogger(ABC):
    SLICE_LOG_PREFIX = 'slice:'

    def create_slice_log_message(self, _slice: Optional[Mapping[str, Any]]) -> AirbyteMessage:
        if False:
            print('Hello World!')
        '\n        Mapping is an interface that can be implemented in various ways. However, json.dumps will just do a `str(<object>)` if\n        the slice is a class implementing Mapping. Therefore, we want to cast this as a dict before passing this to json.dump\n        '
        printable_slice = dict(_slice) if _slice else _slice
        return AirbyteMessage(type=MessageType.LOG, log=AirbyteLogMessage(level=Level.INFO, message=f'{SliceLogger.SLICE_LOG_PREFIX}{json.dumps(printable_slice, default=str)}'))

    @abstractmethod
    def should_log_slice_message(self, logger: logging.Logger) -> bool:
        if False:
            print('Hello World!')
        '\n\n        :param logger:\n        :return:\n        '

class DebugSliceLogger(SliceLogger):

    def should_log_slice_message(self, logger: logging.Logger) -> bool:
        if False:
            print('Hello World!')
        '\n\n        :param logger:\n        :return:\n        '
        return logger.isEnabledFor(logging.DEBUG)

class AlwaysLogSliceLogger(SliceLogger):

    def should_log_slice_message(self, logger: logging.Logger) -> bool:
        if False:
            i = 10
            return i + 15
        return True