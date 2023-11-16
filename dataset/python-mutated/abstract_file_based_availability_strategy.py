import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple
from airbyte_cdk.sources import Source
from airbyte_cdk.sources.streams.availability_strategy import AvailabilityStrategy
from airbyte_cdk.sources.streams.core import Stream
if TYPE_CHECKING:
    from airbyte_cdk.sources.file_based.stream import AbstractFileBasedStream

class AbstractFileBasedAvailabilityStrategy(AvailabilityStrategy):

    @abstractmethod
    def check_availability(self, stream: Stream, logger: logging.Logger, _: Optional[Source]) -> Tuple[bool, Optional[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Perform a connection check for the stream.\n\n        Returns (True, None) if successful, otherwise (False, <error message>).\n        '
        ...

    @abstractmethod
    def check_availability_and_parsability(self, stream: 'AbstractFileBasedStream', logger: logging.Logger, _: Optional[Source]) -> Tuple[bool, Optional[str]]:
        if False:
            print('Hello World!')
        '\n        Performs a connection check for the stream, as well as additional checks that\n        verify that the connection is working as expected.\n\n        Returns (True, None) if successful, otherwise (False, <error message>).\n        '
        ...