from abc import abstractmethod, ABC
from typing import Generic, List, Iterable, Dict, Set
from pyflink.common import Row
from pyflink.common.constants import MAX_LONG_VALUE
from pyflink.datastream.state import MapState
from pyflink.fn_execution.table.window_assigner import WindowAssigner, PanedWindowAssigner, MergingWindowAssigner
from pyflink.fn_execution.table.window_context import Context, K, W

def join_row(left: List, right: List):
    if False:
        i = 10
        return i + 15
    return Row(*left + right)

class InternalWindowProcessFunction(Generic[K, W], ABC):
    """
    The internal interface for functions that process over grouped windows.
    """

    def __init__(self, allowed_lateness: int, window_assigner: WindowAssigner[W], window_aggregator):
        if False:
            return 10
        self._allowed_lateness = allowed_lateness
        self._window_assigner = window_assigner
        self._window_aggregator = window_aggregator
        self._ctx = None

    def open(self, ctx: Context[K, W]):
        if False:
            print('Hello World!')
        self._ctx = ctx
        self._window_assigner.open(ctx)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def is_cleanup_time(self, window: W, time: int) -> bool:
        if False:
            print('Hello World!')
        return time == self._cleanup_time(window)

    def is_window_late(self, window: W) -> bool:
        if False:
            print('Hello World!')
        return self._window_assigner.is_event_time() and self._cleanup_time(window) <= self._ctx.current_watermark()

    def _cleanup_time(self, window: W) -> int:
        if False:
            while True:
                i = 10
        if self._window_assigner.is_event_time():
            cleanup_time = window.max_timestamp() + self._allowed_lateness
            if cleanup_time >= window.max_timestamp():
                return cleanup_time
            else:
                return MAX_LONG_VALUE
        else:
            return window.max_timestamp()

    @abstractmethod
    def assign_state_namespace(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            return 10
        '\n        Assigns the input element into the state namespace which the input element should be\n        accumulated/retracted into.\n\n        :param input_row: The input element\n        :param timestamp: The timestamp of the element or the processing time (depends on the type\n            of assigner)\n        :return: The state namespace.\n        '
        pass

    @abstractmethod
    def assign_actual_windows(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assigns the input element into the actual windows which the {@link Trigger} should trigger\n        on.\n\n        :param input_row: The input element\n        :param timestamp: The timestamp of the element or the processing time (depends on the type\n            of assigner)\n        :return: The actual windows\n        '
        pass

    @abstractmethod
    def prepare_aggregate_accumulator_for_emit(self, window: W):
        if False:
            print('Hello World!')
        '\n        Prepares the accumulator of the given window before emit the final result. The accumulator\n        is stored in the state or will be created if there is no corresponding accumulator in state.\n\n        :param window: The window\n        '
        pass

    @abstractmethod
    def clean_window_if_needed(self, window: W, current_time: int):
        if False:
            return 10
        '\n        Cleans the given window if needed.\n\n        :param window: The window to cleanup\n        :param current_time: The current timestamp\n        '
        pass

class GeneralWindowProcessFunction(InternalWindowProcessFunction[K, W]):
    """
    The general implementation of InternalWindowProcessFunction. The WindowAssigner should be a
    regular assigner without implement PanedWindowAssigner or MergingWindowAssigner.
    """

    def __init__(self, allowed_lateness: int, window_assigner: WindowAssigner[W], window_aggregator):
        if False:
            return 10
        super(GeneralWindowProcessFunction, self).__init__(allowed_lateness, window_assigner, window_aggregator)
        self._reuse_affected_windows = None

    def assign_state_namespace(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            while True:
                i = 10
        element_windows = self._window_assigner.assign_windows(input_row, timestamp)
        self._reuse_affected_windows = []
        for window in element_windows:
            if not self.is_window_late(window):
                self._reuse_affected_windows.append(window)
        return self._reuse_affected_windows

    def assign_actual_windows(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            print('Hello World!')
        return self._reuse_affected_windows

    def prepare_aggregate_accumulator_for_emit(self, window: W):
        if False:
            for i in range(10):
                print('nop')
        acc = self._ctx.get_window_accumulators(window)
        if acc is None:
            acc = self._window_aggregator.create_accumulators()
        self._window_aggregator.set_accumulators(window, acc)

    def clean_window_if_needed(self, window: W, current_time: int):
        if False:
            return 10
        if self.is_cleanup_time(window, current_time):
            self._ctx.clear_window_state(window)
            self._window_aggregator.cleanup(window)
            self._ctx.clear_trigger(window)

class PanedWindowProcessFunction(InternalWindowProcessFunction[K, W]):
    """
    The implementation of InternalWindowProcessFunction for PanedWindowAssigner.
    """

    def __init__(self, allowed_lateness: int, window_assigner: PanedWindowAssigner[W], window_aggregator):
        if False:
            for i in range(10):
                print('nop')
        super(PanedWindowProcessFunction, self).__init__(allowed_lateness, window_assigner, window_aggregator)
        self._window_assigner = window_assigner

    def assign_state_namespace(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            i = 10
            return i + 15
        pane = self._window_assigner.assign_pane(input_row, timestamp)
        if not self._is_pane_late(pane):
            return [pane]
        else:
            return []

    def assign_actual_windows(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            for i in range(10):
                print('nop')
        element_windows = self._window_assigner.assign_windows(input_row, timestamp)
        actual_windows = []
        for window in element_windows:
            if not self.is_window_late(window):
                actual_windows.append(window)
        return actual_windows

    def prepare_aggregate_accumulator_for_emit(self, window: W):
        if False:
            print('Hello World!')
        panes = self._window_assigner.split_into_panes(window)
        acc = self._window_aggregator.create_accumulators()
        self._window_aggregator.set_accumulators(None, acc)
        for pane in panes:
            pane_acc = self._ctx.get_window_accumulators(pane)
            if pane_acc:
                self._window_aggregator.merge(pane, pane_acc)

    def clean_window_if_needed(self, window: W, current_time: int):
        if False:
            return 10
        if self.is_cleanup_time(window, current_time):
            panes = self._window_assigner.split_into_panes(window)
            for pane in panes:
                last_window = self._window_assigner.get_last_window(pane)
                if window == last_window:
                    self._ctx.clear_window_state(pane)
            self._ctx.clear_trigger(window)

    def _is_pane_late(self, pane: W):
        if False:
            i = 10
            return i + 15
        return self._window_assigner.is_event_time() and self.is_window_late(self._window_assigner.get_last_window(pane))

class MergeResultCollector(MergingWindowAssigner.MergeCallback):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.merge_results = {}

    def merge(self, merge_result: W, to_be_merged: Iterable[W]):
        if False:
            i = 10
            return i + 15
        self.merge_results[merge_result] = to_be_merged

class MergingWindowProcessFunction(InternalWindowProcessFunction[K, W]):
    """
    The implementation of InternalWindowProcessFunction for MergingWindowAssigner.
    """

    def __init__(self, allowed_lateness: int, window_assigner: MergingWindowAssigner[W], window_aggregator, state_backend):
        if False:
            for i in range(10):
                print('nop')
        super(MergingWindowProcessFunction, self).__init__(allowed_lateness, window_assigner, window_aggregator)
        self._window_assigner = window_assigner
        self._reuse_actual_windows = None
        self._window_mapping = None
        self._state_backend = state_backend
        self._sorted_windows = None
        from pyflink.fn_execution.state_impl import LRUCache
        self._cached_sorted_windows = LRUCache(10000, None)

    def open(self, ctx: Context[K, W]):
        if False:
            while True:
                i = 10
        super(MergingWindowProcessFunction, self).open(ctx)
        self._window_mapping = self._state_backend.get_map_state('session-window-mapping', self._state_backend.namespace_coder, self._state_backend.namespace_coder)

    def assign_state_namespace(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            while True:
                i = 10
        element_windows = self._window_assigner.assign_windows(input_row, timestamp)
        self._initialize_cache(self._ctx.current_key())
        self._reuse_actual_windows = []
        for window in element_windows:
            actual_window = self._add_window(window)
            if self.is_window_late(actual_window):
                self._window_mapping.remove(actual_window)
                self._sorted_windows.remove(actual_window)
            else:
                self._reuse_actual_windows.append(actual_window)
        affected_windows = [self._window_mapping.get(actual) for actual in self._reuse_actual_windows]
        return affected_windows

    def assign_actual_windows(self, input_row: List, timestamp: int) -> List[W]:
        if False:
            i = 10
            return i + 15
        return self._reuse_actual_windows

    def prepare_aggregate_accumulator_for_emit(self, window: W):
        if False:
            return 10
        state_window = self._window_mapping.get(window)
        acc = self._ctx.get_window_accumulators(state_window)
        if acc is None:
            acc = self._window_aggregator.create_accumulators()
        self._window_aggregator.set_accumulators(state_window, acc)

    def clean_window_if_needed(self, window: W, current_time: int):
        if False:
            print('Hello World!')
        if self.is_cleanup_time(window, current_time):
            self._ctx.clear_trigger(window)
            state_window = self._window_mapping.get(window)
            self._ctx.clear_window_state(state_window)
            self._initialize_cache(self._ctx.current_key())
            self._window_mapping.remove(window)
            self._sorted_windows.remove(window)

    def _initialize_cache(self, key):
        if False:
            return 10
        tuple_key = tuple(key)
        self._sorted_windows = self._cached_sorted_windows.get(tuple_key)
        if self._sorted_windows is None:
            self._sorted_windows = [k for k in self._window_mapping]
            self._sorted_windows.sort()
            self._cached_sorted_windows.put(tuple_key, self._sorted_windows)

    def _add_window(self, new_window: W):
        if False:
            while True:
                i = 10
        collector = MergeResultCollector()
        self._window_assigner.merge_windows(new_window, self._sorted_windows, collector)
        result_window = new_window
        is_new_window_merged = False
        merge_results = collector.merge_results
        for merge_result in merge_results:
            merge_windows = merge_results[merge_result]
            try:
                merge_windows.remove(new_window)
                is_new_window_merged = True
                result_window = merge_result
            except KeyError:
                pass
            if not merge_windows:
                continue
            merged_state_namespace = self._window_mapping.get(iter(merge_windows).__next__())
            merged_state_windows = []
            for merged_window in merge_windows:
                res = self._window_mapping.get(merged_window)
                if res is not None:
                    self._window_mapping.remove(merged_window)
                    self._sorted_windows.remove(merged_window)
                    if res != merged_state_namespace:
                        merged_state_windows.append(res)
            self._window_mapping.put(merge_result, merged_state_namespace)
            self._sorted_windows.append(merge_result)
            self._sorted_windows.sort()
            if not (len(merge_windows) == 1 and merge_result in merge_windows):
                self._merge(merge_result, merge_windows, merged_state_namespace, merged_state_windows)
        if len(merge_results) == 0 or (result_window == new_window and (not is_new_window_merged)):
            self._window_mapping.put(result_window, result_window)
            self._sorted_windows.append(result_window)
            self._sorted_windows.sort()
        return result_window

    def _merge(self, merge_result: W, merge_windows: Set[W], state_window_result: W, state_windows_tobe_merged: Iterable[W]):
        if False:
            while True:
                i = 10
        self._ctx.on_merge(merge_result, state_windows_tobe_merged)
        for window in merge_windows:
            self._ctx.clear_trigger(window)
            self._ctx.delete_cleanup_timer(window)
        if state_windows_tobe_merged:
            target_acc = self._ctx.get_window_accumulators(state_window_result)
            if target_acc is None:
                target_acc = self._window_aggregator.create_accumulators()
            self._window_aggregator.set_accumulators(state_window_result, target_acc)
            for window in state_windows_tobe_merged:
                acc = self._ctx.get_window_accumulators(window)
                if acc is not None:
                    self._window_aggregator.merge(window, acc)
                self._ctx.clear_window_state(window)
            target_acc = self._window_aggregator.get_accumulators()
            self._ctx.set_window_accumulators(state_window_result, target_acc)