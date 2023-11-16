"""
Interfaces and generic implementations for observability events (like CW logs)
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union
from samcli.lib.utils.async_utils import AsyncContext
LOG = logging.getLogger(__name__)
InternalEventType = TypeVar('InternalEventType')

class ObservabilityEvent(Generic[InternalEventType]):
    """
    Generic class that represents observability event
    This keeps some common fields for filtering or sorting later on
    """

    def __init__(self, event: InternalEventType, timestamp: int, resource_name: Optional[str]=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        event : EventType\n            Actual event object. This can be any type with generic definition (dict, str etc.)\n        timestamp : int\n            Timestamp of the event\n        resource_name : Optional[str]\n            Resource name related to this event. This is optional since not all events is connected to a single resource\n        '
        self.event = event
        self.timestamp = timestamp
        self.resource_name = resource_name
ObservabilityEventType = TypeVar('ObservabilityEventType', bound=ObservabilityEvent)

class ObservabilityPuller(ABC):
    """
    Interface definition for pulling observability information.
    """
    cancelled: bool = False

    @abstractmethod
    def tail(self, start_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        start_time : Optional[datetime]\n            Optional parameter to tail information from earlier time\n        filter_pattern :  Optional[str]\n            Optional parameter to filter events with given string\n        '

    @abstractmethod
    def load_time_period(self, start_time: Optional[datetime]=None, end_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        start_time : Optional[datetime]\n            Optional parameter to load events from certain date time\n        end_time :  Optional[datetime]\n            Optional parameter to load events until certain date time\n        filter_pattern : Optional[str]\n            Optional parameter to filter events with given string\n        '

    @abstractmethod
    def load_events(self, event_ids: Union[List[Any], Dict]):
        if False:
            i = 10
            return i + 15
        '\n        This method will load specific events which is given by the event_ids parameter\n\n        Parameters\n        ----------\n        event_ids : List[str] or Dict\n            List of event ids that will be pulled\n        '

    def stop_tailing(self):
        if False:
            for i in range(10):
                print('nop')
        self.cancelled = True

class ObservabilityEventMapper(Generic[ObservabilityEventType]):
    """
    Interface definition to map/change any event to another object
    This could be used by highlighting certain parts or formatting events before logging into console
    """

    @abstractmethod
    def map(self, event: ObservabilityEventType) -> Any:
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        event : ObservabilityEventType\n            Event object that will be mapped/converted to another event or any object\n\n        Returns\n        -------\n        Any\n            Return converted type\n        '

class ObservabilityEventConsumer(Generic[ObservabilityEventType]):
    """
    Consumer interface, which will consume any event.
    An example is to output event into console.
    """

    @abstractmethod
    def consume(self, event: ObservabilityEventType):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        event : ObservabilityEvent\n            Event that will be consumed\n        '

class ObservabilityEventConsumerDecorator(ObservabilityEventConsumer):
    """
    A decorator implementation for consumer, which can have mappers and decorated consumer within.
    Rather than the normal implementation, this will process the events through mappers which is been
    provided, and then pass them to actual consumer
    """

    def __init__(self, mappers: List[ObservabilityEventMapper], consumer: ObservabilityEventConsumer):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        mappers : List[ObservabilityEventMapper]\n            List of event mappers which will be used to process events before passing to consumer\n        consumer : ObservabilityEventConsumer\n            Actual consumer which will handle the events after they are processed by mappers\n        '
        super().__init__()
        self._mappers = mappers
        self._consumer = consumer

    def consume(self, event: ObservabilityEvent):
        if False:
            return 10
        '\n        See Also ObservabilityEventConsumerDecorator and ObservabilityEventConsumer\n        '
        for mapper in self._mappers:
            LOG.debug('Calling mapper (%s) for event (%s)', mapper, event)
            event = mapper.map(event)
        LOG.debug('Calling consumer (%s) for event (%s)', self._consumer, event)
        self._consumer.consume(event)

class ObservabilityCombinedPuller(ObservabilityPuller):
    """
    A decorator class which will contain multiple ObservabilityPuller instance and pull information from each of them
    """

    def __init__(self, pullers: Sequence[ObservabilityPuller]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        pullers : List[ObservabilityPuller]\n            List of pullers which will be managed by this class\n        '
        self._pullers = pullers

    def tail(self, start_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            while True:
                i = 10
        '\n        Implementation of ObservabilityPuller.tail method with AsyncContext.\n        It will create tasks by calling tail methods of all given pullers, and execute them in async\n        '
        async_context = AsyncContext()
        for puller in self._pullers:
            LOG.debug("Adding task 'tail' for puller (%s)", puller)
            async_context.add_async_task(puller.tail, start_time, filter_pattern)
        LOG.debug("Running all 'tail' tasks in parallel")
        try:
            async_context.run_async()
        except KeyboardInterrupt:
            LOG.info(' CTRL+C received, cancelling...')
            self.stop_tailing()

    def load_time_period(self, start_time: Optional[datetime]=None, end_time: Optional[datetime]=None, filter_pattern: Optional[str]=None):
        if False:
            while True:
                i = 10
        '\n        Implementation of ObservabilityPuller.load_time_period method with AsyncContext.\n        It will create tasks by calling load_time_period methods of all given pullers, and execute them in async\n        '
        async_context = AsyncContext()
        for puller in self._pullers:
            LOG.debug("Adding task 'load_time_period' for puller (%s)", puller)
            async_context.add_async_task(puller.load_time_period, start_time, end_time, filter_pattern)
        LOG.debug("Running all 'load_time_period' tasks in parallel")
        async_context.run_async()

    def load_events(self, event_ids: Union[List[Any], Dict]):
        if False:
            while True:
                i = 10
        '\n        Implementation of ObservabilityPuller.load_events method with AsyncContext.\n        It will create tasks by calling load_events methods of all given pullers, and execute them in async\n        '
        async_context = AsyncContext()
        for puller in self._pullers:
            LOG.debug("Adding task 'load_events' for puller (%s)", puller)
            async_context.add_async_task(puller.load_events, event_ids)
        LOG.debug("Running all 'load_time_period' tasks in parallel")
        async_context.run_async()

    def stop_tailing(self):
        if False:
            return 10
        for puller in self._pullers:
            puller.stop_tailing()