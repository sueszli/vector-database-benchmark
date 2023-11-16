import threading
from typing import Any, Callable, Dict, List, Optional, Set
from streamlit.proto.WidgetStates_pb2 import WidgetState as WidgetStateProto
from streamlit.proto.WidgetStates_pb2 import WidgetStates as WidgetStatesProto
from streamlit.runtime.state.common import RegisterWidgetResult, T, WidgetMetadata
from streamlit.runtime.state.session_state import SessionState

class SafeSessionState:
    """Thread-safe wrapper around SessionState.

    When AppSession gets a re-run request, it can interrupt its existing
    ScriptRunner and spin up a new ScriptRunner to handle the request.
    When this happens, the existing ScriptRunner will continue executing
    its script until it reaches a yield point - but during this time, it
    must not mutate its SessionState.
    """
    _state: SessionState
    _lock: threading.RLock
    _yield_callback: Callable[[], None]

    def __init__(self, state: SessionState, yield_callback: Callable[[], None]):
        if False:
            for i in range(10):
                print('nop')
        object.__setattr__(self, '_state', state)
        object.__setattr__(self, '_lock', threading.RLock())
        object.__setattr__(self, '_yield_callback', yield_callback)

    def register_widget(self, metadata: WidgetMetadata[T], user_key: Optional[str]) -> RegisterWidgetResult[T]:
        if False:
            i = 10
            return i + 15
        self._yield_callback()
        with self._lock:
            return self._state.register_widget(metadata, user_key)

    def on_script_will_rerun(self, latest_widget_states: WidgetStatesProto) -> None:
        if False:
            i = 10
            return i + 15
        self._yield_callback()
        with self._lock:
            self._state.on_script_will_rerun(latest_widget_states)

    def on_script_finished(self, widget_ids_this_run: Set[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._state.on_script_finished(widget_ids_this_run)

    def maybe_check_serializable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._state.maybe_check_serializable()

    def get_widget_states(self) -> List[WidgetStateProto]:
        if False:
            print('Hello World!')
        'Return a list of serialized widget values for each widget with a value.'
        with self._lock:
            return self._state.get_widget_states()

    def is_new_state_value(self, user_key: str) -> bool:
        if False:
            while True:
                i = 10
        with self._lock:
            return self._state.is_new_state_value(user_key)

    @property
    def filtered_state(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'The combined session and widget state, excluding keyless widgets.'
        with self._lock:
            return self._state.filtered_state

    def __getitem__(self, key: str) -> Any:
        if False:
            while True:
                i = 10
        self._yield_callback()
        with self._lock:
            return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._yield_callback()
        with self._lock:
            self._state[key] = value

    def __delitem__(self, key: str) -> None:
        if False:
            return 10
        self._yield_callback()
        with self._lock:
            del self._state[key]

    def __contains__(self, key: str) -> bool:
        if False:
            return 10
        self._yield_callback()
        with self._lock:
            return key in self._state

    def __getattr__(self, key: str) -> Any:
        if False:
            print('Hello World!')
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f'{key} not found in session_state.')

    def __setattr__(self, key: str, value: Any) -> None:
        if False:
            print('Hello World!')
        self[key] = value

    def __delattr__(self, key: str) -> None:
        if False:
            print('Hello World!')
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f'{key} not found in session_state.')

    def __repr__(self):
        if False:
            return 10
        'Presents itself as a simple dict of the underlying SessionState instance'
        kv = ((k, self._state[k]) for k in self._state._keys())
        s = ', '.join((f'{k}: {v!r}' for (k, v) in kv))
        return f'{{{s}}}'