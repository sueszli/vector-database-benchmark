import logging
import typing
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from airbyte_cdk.sources.streams import Stream
if typing.TYPE_CHECKING:
    from airbyte_cdk.sources import Source

class AvailabilityStrategy(ABC):
    """
    Abstract base class for checking stream availability.
    """

    @abstractmethod
    def check_availability(self, stream: Stream, logger: logging.Logger, source: Optional['Source']) -> Tuple[bool, Optional[str]]:
        if False:
            return 10
        '\n        Checks stream availability.\n\n        :param stream: stream\n        :param logger: source logger\n        :param source: (optional) source\n        :return: A tuple of (boolean, str). If boolean is true, then the stream\n          is available, and no str is required. Otherwise, the stream is unavailable\n          for some reason and the str should describe what went wrong and how to\n          resolve the unavailability, if possible.\n        '