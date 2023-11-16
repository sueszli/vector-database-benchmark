import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states

class ScriptRequestType(Enum):
    CONTINUE = 'CONTINUE'
    STOP = 'STOP'
    RERUN = 'RERUN'

@dataclass(frozen=True)
class RerunData:
    """Data attached to RERUN requests. Immutable."""
    query_string: str = ''
    widget_states: Optional[WidgetStates] = None
    page_script_hash: str = ''
    page_name: str = ''

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return util.repr_(self)

@dataclass(frozen=True)
class ScriptRequest:
    """A STOP or RERUN request and associated data."""
    type: ScriptRequestType
    _rerun_data: Optional[RerunData] = None

    @property
    def rerun_data(self) -> RerunData:
        if False:
            print('Hello World!')
        if self.type is not ScriptRequestType.RERUN:
            raise RuntimeError('RerunData is only set for RERUN requests.')
        return cast(RerunData, self._rerun_data)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return util.repr_(self)

class ScriptRequests:
    """An interface for communicating with a ScriptRunner. Thread-safe.

    AppSession makes requests of a ScriptRunner through this class, and
    ScriptRunner handles those requests.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._lock = threading.Lock()
        self._state = ScriptRequestType.CONTINUE
        self._rerun_data = RerunData()

    def request_stop(self) -> None:
        if False:
            i = 10
            return i + 15
        "Request that the ScriptRunner stop running. A stopped ScriptRunner\n        can't be used anymore. STOP requests succeed unconditionally.\n        "
        with self._lock:
            self._state = ScriptRequestType.STOP

    def request_rerun(self, new_data: RerunData) -> bool:
        if False:
            print('Hello World!')
        "Request that the ScriptRunner rerun its script.\n\n        If the ScriptRunner has been stopped, this request can't be honored:\n        return False.\n\n        Otherwise, record the request and return True. The ScriptRunner will\n        handle the rerun request as soon as it reaches an interrupt point.\n        "
        with self._lock:
            if self._state == ScriptRequestType.STOP:
                return False
            if self._state == ScriptRequestType.CONTINUE:
                self._state = ScriptRequestType.RERUN
                self._rerun_data = new_data
                return True
            if self._state == ScriptRequestType.RERUN:
                if self._rerun_data.widget_states is None:
                    self._rerun_data = new_data
                    return True
                if new_data.widget_states is not None:
                    coalesced_states = coalesce_widget_states(self._rerun_data.widget_states, new_data.widget_states)
                    self._rerun_data = RerunData(query_string=new_data.query_string, widget_states=coalesced_states, page_script_hash=new_data.page_script_hash, page_name=new_data.page_name)
                    return True
                return True
            raise RuntimeError(f'Unrecognized ScriptRunnerState: {self._state}')

    def on_scriptrunner_yield(self) -> Optional[ScriptRequest]:
        if False:
            for i in range(10):
                print('nop')
        "Called by the ScriptRunner when it's at a yield point.\n\n        If we have no request, return None.\n\n        If we have a RERUN request, return the request and set our internal\n        state to CONTINUE.\n\n        If we have a STOP request, return the request and remain stopped.\n        "
        if self._state == ScriptRequestType.CONTINUE:
            return None
        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
            assert self._state == ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)

    def on_scriptrunner_ready(self) -> ScriptRequest:
        if False:
            i = 10
            return i + 15
        "Called by the ScriptRunner when it's about to run its script for\n        the first time, and also after its script has successfully completed.\n\n        If we have a RERUN request, return the request and set\n        our internal state to CONTINUE.\n\n        If we have a STOP request or no request, set our internal state\n        to STOP.\n        "
        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
            self._state = ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)