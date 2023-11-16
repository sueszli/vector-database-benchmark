from abc import ABC, abstractmethod

class TimerService(ABC):
    """
    Interface for working with time and timers.
    """

    @abstractmethod
    def current_processing_time(self):
        if False:
            print('Hello World!')
        '\n        Returns the current processing time.\n        '
        pass

    @abstractmethod
    def current_watermark(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the current event-time watermark.\n        '
        pass

    @abstractmethod
    def register_processing_time_timer(self, timestamp: int):
        if False:
            i = 10
            return i + 15
        '\n        Registers a timer to be fired when processing time passes the given time.\n\n        Timers can internally be scoped to keys and/or windows. When you set a timer in a keyed\n        context, such as in an operation on KeyedStream then that context will so be active when you\n        receive the timer notification.\n\n        :param timestamp: The processing time of the timer to be registered.\n        '
        pass

    @abstractmethod
    def register_event_time_timer(self, timestamp: int):
        if False:
            print('Hello World!')
        '\n        Registers a timer tobe fired when the event time watermark passes the given time.\n\n        Timers can internally be scoped to keys and/or windows. When you set a timer in a keyed\n        context, such as in an operation on KeyedStream then that context will so be active when you\n        receive the timer notification.\n\n        :param timestamp: The event time of the timer to be registered.\n        '
        pass

    def delete_processing_time_timer(self, timestamp: int):
        if False:
            while True:
                i = 10
        '\n        Deletes the processing-time timer with the given trigger time. This method has only an\n        effect if such a timer was previously registered and did not already expire.\n\n        Timers can internally be scoped to keys and/or windows. When you delete a timer, it is\n        removed from the current keyed context.\n\n        :param timestamp: The given trigger time of timer to be deleted.\n        '
        pass

    def delete_event_time_timer(self, timestamp: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes the event-time timer with the given trigger time. This method has only an effect if\n        such a timer was previously registered and did not already expire.\n\n        Timers can internally be scoped to keys and/or windows. When you delete a timer, it is\n        removed from the current keyed context.\n\n        :param timestamp: The given trigger time of timer to be deleted.\n        '
        pass