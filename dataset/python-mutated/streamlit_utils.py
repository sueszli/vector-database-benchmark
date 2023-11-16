"""Hack to add per-session state to Streamlit.

Based on: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
"""
try:
    import streamlit.ReportThread as ReportThread
    from streamlit.ReportSession import ReportSession
    from streamlit.server.Server import Server
except Exception:
    import streamlit.report_thread as ReportThread
    from streamlit.server.server import Server
    from streamlit.report_session import ReportSession
from typing import Any, Dict
CUSTOM_STREAMLIT_CSS = '\ndiv[data-testid="stBlock"] button {\n  width: 100% !important;\n  margin-bottom: 20px !important;\n  border-color: #bfbfbf !important;\n}\npre code {\n    white-space: pre-wrap;\n}\n'

class SessionState(object):

    def __init__(self, **kwargs: Any) -> None:
        if False:
            return 10
        'A new SessionState object.'
        self._run_id = 0
        self._input_data: Dict = {}
        self._output_data: Any = None
        self._latest_operation_input: Any = None
        for (key, val) in kwargs.items():
            setattr(self, key, val)

    @property
    def run_id(self) -> int:
        if False:
            return 10
        return self._run_id

    @property
    def input_data(self) -> Dict:
        if False:
            return 10
        return self._input_data

    @property
    def output_data(self) -> Any:
        if False:
            print('Hello World!')
        return self._output_data

    @output_data.setter
    def output_data(self, output_data: Any) -> None:
        if False:
            print('Hello World!')
        self._output_data = output_data

    @property
    def latest_operation_input(self) -> Any:
        if False:
            while True:
                i = 10
        return self._latest_operation_input

    @latest_operation_input.setter
    def latest_operation_input(self, latest_operation_input: Any) -> None:
        if False:
            print('Hello World!')
        self._latest_operation_input = latest_operation_input

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._run_id += 1
        self._input_data = {}
        self._output_data = None
        self._latest_operation_input = None

def get_current_session() -> ReportSession:
    if False:
        return 10
    ctx = ReportThread.get_report_ctx()
    this_session = None
    current_server = Server.get_current()
    if hasattr(current_server, '_session_infos'):
        session_infos = Server.get_current()._session_infos.values()
    else:
        session_infos = Server.get_current()._session_info_by_id.values()
    for session_info in session_infos:
        s = session_info.session
        if hasattr(s, '_main_dg') and s._main_dg == ctx.main_dg or (not hasattr(s, '_main_dg') and s.enqueue == ctx.enqueue) or (not hasattr(s, '_main_dg') and s._uploaded_file_mgr == ctx.uploaded_file_mgr):
            this_session = s
    if this_session is None:
        raise RuntimeError("Oh noes. Couldn't get your Streamlit Session object. Are you doing something fancy with threads?")
    return this_session

def get_session_state(**kwargs: Any) -> SessionState:
    if False:
        print('Hello World!')
    'Gets a SessionState object for the current session.\n\n    Creates a new object if necessary.\n    '
    this_session = get_current_session()
    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)
    return this_session._custom_session_state