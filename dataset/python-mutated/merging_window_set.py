from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Collection, Iterable
from pyflink.datastream import MergingWindowAssigner
from pyflink.datastream.state import MapState
W = TypeVar('W')

class MergeResultsCallback(MergingWindowAssigner.MergeCallback):

    def __init__(self, merge_results: dict):
        if False:
            i = 10
            return i + 15
        self._merge_results = merge_results

    def merge(self, to_be_merged: Iterable[W], merge_result: W) -> None:
        if False:
            i = 10
            return i + 15
        self._merge_results[merge_result] = to_be_merged

class MergingWindowSet(Generic[W]):

    class MergeFunction(ABC, Generic[W]):

        @abstractmethod
        def merge(self, merge_result: W, merged_windows: Collection[W], state_window_result: W, merged_state_windows: Collection[W]):
            if False:
                print('Hello World!')
            pass

    def __init__(self, assigner: MergingWindowAssigner, state: MapState[W, W]):
        if False:
            return 10
        self._window_assigner = assigner
        self._mapping = dict()
        for (window_for_user, window_in_state) in state.items():
            self._mapping[window_for_user] = window_in_state
        self._state = state
        self._initial_mapping = dict(self._mapping)

    def persist(self) -> None:
        if False:
            print('Hello World!')
        if self._mapping != self._initial_mapping:
            self._state.clear()
            for (window_for_user, window_in_state) in self._mapping.items():
                self._state.put(window_for_user, window_in_state)

    def get_state_window(self, window: W) -> W:
        if False:
            print('Hello World!')
        if window in self._mapping:
            return self._mapping[window]
        else:
            return None

    def retire_window(self, window) -> None:
        if False:
            for i in range(10):
                print('nop')
        if window in self._mapping:
            self._mapping.pop(window)
        else:
            raise Exception('Window %s is not in in-flight window set.' % window)

    def add_window(self, new_window: W, merge_function: MergeFunction[W]):
        if False:
            for i in range(10):
                print('nop')
        windows = []
        windows.extend(self._mapping.keys())
        windows.append(new_window)
        merge_results = dict()
        self._window_assigner.merge_windows(windows, MergeResultsCallback(merge_results))
        result_window = new_window
        merged_new_window = False
        for (merge_result, merged_windows) in merge_results.items():
            if new_window in merged_windows:
                merged_new_window = True
                merged_windows.remove(new_window)
                result_window = merge_result
            merged_state_window = self._mapping[next(iter(merged_windows))]
            merged_state_windows = []
            for merged_window in merged_windows:
                if merged_window in self._mapping:
                    res = self._mapping.pop(merged_window)
                    merged_state_windows.append(res)
            self._mapping[merge_result] = merged_state_window
            merged_state_windows.remove(merged_state_window)
            if merge_result not in merged_windows or len(merged_windows) != 1:
                merge_function.merge(merge_result, merged_windows, self._mapping[merge_result], merged_state_windows)
        if len(merge_results) == 0 or (result_window == new_window and (not merged_new_window)):
            self._mapping[result_window] = result_window
        return result_window