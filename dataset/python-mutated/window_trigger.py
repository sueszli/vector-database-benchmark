from abc import abstractmethod, ABC
from typing import Generic
from pyflink.common.typeinfo import Types
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.datastream.window import TimeWindow, CountWindow
from pyflink.fn_execution.table.window_context import TriggerContext, W

class Trigger(Generic[W], ABC):
    """
    A Trigger determines when a pane of a window should be evaluated to emit the results for
    that part of the window.

    A pane is the bucket of elements that have the same key and same Window. An element
    an be in multiple panes if it was assigned to multiple windows by the WindowAssigner.
    These panes all have their own instance of the Trigger.

    Triggers must not maintain state internally since they can be re-created or reused for
    different keys. All necessary state should be persisted using the state abstraction available on
    the TriggerContext.
    """

    @abstractmethod
    def open(self, ctx: TriggerContext):
        if False:
            while True:
                i = 10
        '\n        Initialization method for the trigger. Creates states in this method.\n\n        :param ctx: A context object that can be used to get states.\n        '
        pass

    @abstractmethod
    def on_element(self, element, timestamp, window: W) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Called for every element that gets added to a pane. The result of this will determine\n        whether the pane is evaluated to emit results.\n\n        :param element: The element that arrived.\n        :param timestamp: The timestamp of the element that arrived.\n        :param window: The window to which the element is being added.\n        :return: True for firing the window, False for no action\n        '
        pass

    @abstractmethod
    def on_processing_time(self, time: int, window: W) -> bool:
        if False:
            while True:
                i = 10
        '\n        Called when a processing-time timer that was set using the trigger context fires.\n\n        This method is not called in case the window does not contain any elements. Thus, if\n        you return PURGE from a trigger method and you expect to do cleanup in a future\n        invocation of a timer callback it might be wise to clean any state that you would clean in\n        the timer callback.\n\n        :param time: The timestamp at which the timer fired.\n        :param window: The window for which the timer fired.\n        :return: True for firing the window, False for no action\n        '
        pass

    @abstractmethod
    def on_event_time(self, time: int, window: W) -> bool:
        if False:
            print('Hello World!')
        '\n        Called when a event-time timer that was set using the trigger context fires.\n\n        This method is not called in case the window does not contain any elements. Thus, if\n        you return PURGE from a trigger method and you expect to do cleanup in a future\n        invocation of a timer callback it might be wise to clean any state that you would clean in\n        the timer callback.\n\n        :param time: The timestamp at which the timer fired.\n        :param window: The window for which the timer fired.\n        :return: True for firing the window, False for no action\n        '
        pass

    @abstractmethod
    def on_merge(self, window: W, merge_context: TriggerContext):
        if False:
            i = 10
            return i + 15
        '\n        Called when several windows have been merged into one window by the WindowAssigner.\n        '
        pass

    @abstractmethod
    def clear(self, window: W) -> None:
        if False:
            while True:
                i = 10
        '\n        Clears any state that the trigger might still hold for the given window. This is called when\n        a window is purged. Timers set using TriggerContext.register_event_time_timer(int) and\n        TriggerContext.register_processing_time_timer(int) should be deleted here as well as\n        state acquired using TriggerContext.get_partitioned_state(StateDescriptor).\n        '
        pass

class ProcessingTimeTrigger(Trigger[TimeWindow]):
    """
    A Trigger that fires once the current system time passes the end of the window to which a
    pane belongs.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._ctx = None

    def open(self, ctx: TriggerContext):
        if False:
            while True:
                i = 10
        self._ctx = ctx

    def on_element(self, element, timestamp, window: W) -> bool:
        if False:
            i = 10
            return i + 15
        self._ctx.register_processing_time_timer(window.max_timestamp())
        return False

    def on_processing_time(self, time: int, window: W) -> bool:
        if False:
            i = 10
            return i + 15
        return time == window.max_timestamp()

    def on_event_time(self, time: int, window: W) -> bool:
        if False:
            while True:
                i = 10
        return False

    def on_merge(self, window: W, merge_context: TriggerContext):
        if False:
            while True:
                i = 10
        self._ctx.register_processing_time_timer(window.max_timestamp())

    def clear(self, window: W) -> None:
        if False:
            i = 10
            return i + 15
        self._ctx.delete_processing_time_timer(window.max_timestamp())

class EventTimeTrigger(Trigger[TimeWindow]):
    """
    A Trigger that fires once the watermark passes the end of the window to which a pane
    belongs.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._ctx = None

    def open(self, ctx: TriggerContext):
        if False:
            while True:
                i = 10
        self._ctx = ctx

    def on_element(self, element, timestamp, window: W) -> bool:
        if False:
            print('Hello World!')
        if window.max_timestamp() <= self._ctx.get_current_watermark():
            return True
        else:
            self._ctx.register_event_time_timer(window.max_timestamp())
            return False

    def on_processing_time(self, time: int, window: W) -> bool:
        if False:
            i = 10
            return i + 15
        return False

    def on_event_time(self, time: int, window: W) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return time == window.max_timestamp()

    def on_merge(self, window: W, merge_context: TriggerContext):
        if False:
            print('Hello World!')
        self._ctx.register_event_time_timer(window.max_timestamp())

    def clear(self, window: W) -> None:
        if False:
            while True:
                i = 10
        self._ctx.delete_event_time_timer(window.max_timestamp())

class CountTrigger(Trigger[CountWindow]):
    """
    A Trigger that fires once the count of elements in a pane reaches the given count.
    """

    def __init__(self, count_elements: int):
        if False:
            return 10
        self._count_elements = count_elements
        self._count_state_desc = ValueStateDescriptor('trigger-count-%s' % count_elements, Types.LONG())
        self._ctx = None

    def open(self, ctx: TriggerContext):
        if False:
            i = 10
            return i + 15
        self._ctx = ctx

    def on_element(self, element, timestamp, window: W) -> bool:
        if False:
            for i in range(10):
                print('nop')
        count_state = self._ctx.get_partitioned_state(self._count_state_desc)
        count = count_state.value()
        if count is None:
            count = 0
        count += 1
        count_state.update(count)
        if count >= self._count_elements:
            count_state.clear()
            return True
        else:
            return False

    def on_processing_time(self, time: int, window: W) -> bool:
        if False:
            while True:
                i = 10
        return False

    def on_event_time(self, time: int, window: W) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

    def on_merge(self, window: W, merge_context: TriggerContext):
        if False:
            print('Hello World!')
        merge_context.merge_partitioned_state(self._count_state_desc)

    def clear(self, window: W) -> None:
        if False:
            print('Hello World!')
        self._ctx.get_partitioned_state(self._count_state_desc).clear()