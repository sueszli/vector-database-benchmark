import collections
import threading
from dataclasses import dataclass, field
from typing import Callable, Counter, Dict, List, Optional, Set
from typing_extensions import Final, TypeAlias
from streamlit import runtime
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Command
from streamlit.runtime.scriptrunner.script_requests import ScriptRequests
from streamlit.runtime.state import SafeSessionState
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
LOGGER: Final = get_logger(__name__)
UserInfo: TypeAlias = Dict[str, Optional[str]]

@dataclass
class ScriptRunContext:
    """A context object that contains data for a "script run" - that is,
    data that's scoped to a single ScriptRunner execution (and therefore also
    scoped to a single connected "session").

    ScriptRunContext is used internally by virtually every `st.foo()` function.
    It is accessed only from the script thread that's created by ScriptRunner.

    Streamlit code typically retrieves the active ScriptRunContext via the
    `get_script_run_ctx` function.
    """
    session_id: str
    _enqueue: Callable[[ForwardMsg], None]
    query_string: str
    session_state: SafeSessionState
    uploaded_file_mgr: UploadedFileManager
    page_script_hash: str
    user_info: UserInfo
    gather_usage_stats: bool = False
    command_tracking_deactivated: bool = False
    tracked_commands: List[Command] = field(default_factory=list)
    tracked_commands_counter: Counter[str] = field(default_factory=collections.Counter)
    _set_page_config_allowed: bool = True
    _has_script_started: bool = False
    widget_ids_this_run: Set[str] = field(default_factory=set)
    widget_user_keys_this_run: Set[str] = field(default_factory=set)
    form_ids_this_run: Set[str] = field(default_factory=set)
    cursors: Dict[int, 'streamlit.cursor.RunningCursor'] = field(default_factory=dict)
    dg_stack: List['streamlit.delta_generator.DeltaGenerator'] = field(default_factory=list)
    script_requests: Optional[ScriptRequests] = None

    def reset(self, query_string: str='', page_script_hash: str='') -> None:
        if False:
            while True:
                i = 10
        self.cursors = {}
        self.widget_ids_this_run = set()
        self.widget_user_keys_this_run = set()
        self.form_ids_this_run = set()
        self.query_string = query_string
        self.page_script_hash = page_script_hash
        self._set_page_config_allowed = True
        self._has_script_started = False
        self.command_tracking_deactivated: bool = False
        self.tracked_commands = []
        self.tracked_commands_counter = collections.Counter()

    def on_script_start(self) -> None:
        if False:
            return 10
        self._has_script_started = True

    def enqueue(self, msg: ForwardMsg) -> None:
        if False:
            i = 10
            return i + 15
        "Enqueue a ForwardMsg for this context's session."
        if msg.HasField('page_config_changed') and (not self._set_page_config_allowed):
            raise StreamlitAPIException('`set_page_config()` can only be called once per app page, ' + 'and must be called as the first Streamlit command in your script.\n\n' + 'For more information refer to the [docs]' + '(https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config).')
        if msg.HasField('page_config_changed') or (msg.HasField('delta') and self._has_script_started):
            self._set_page_config_allowed = False
        self._enqueue(msg)
SCRIPT_RUN_CONTEXT_ATTR_NAME: Final = 'streamlit_script_run_ctx'

def add_script_run_ctx(thread: Optional[threading.Thread]=None, ctx: Optional[ScriptRunContext]=None):
    if False:
        for i in range(10):
            print('nop')
    "Adds the current ScriptRunContext to a newly-created thread.\n\n    This should be called from this thread's parent thread,\n    before the new thread starts.\n\n    Parameters\n    ----------\n    thread : threading.Thread\n        The thread to attach the current ScriptRunContext to.\n    ctx : ScriptRunContext or None\n        The ScriptRunContext to add, or None to use the current thread's\n        ScriptRunContext.\n\n    Returns\n    -------\n    threading.Thread\n        The same thread that was passed in, for chaining.\n\n    "
    if thread is None:
        thread = threading.current_thread()
    if ctx is None:
        ctx = get_script_run_ctx()
    if ctx is not None:
        setattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, ctx)
    return thread

def get_script_run_ctx(suppress_warning: bool=False) -> Optional[ScriptRunContext]:
    if False:
        i = 10
        return i + 15
    "\n    Parameters\n    ----------\n    suppress_warning : bool\n        If True, don't log a warning if there's no ScriptRunContext.\n    Returns\n    -------\n    ScriptRunContext | None\n        The current thread's ScriptRunContext, or None if it doesn't have one.\n\n    "
    thread = threading.current_thread()
    ctx: Optional[ScriptRunContext] = getattr(thread, SCRIPT_RUN_CONTEXT_ATTR_NAME, None)
    if ctx is None and runtime.exists() and (not suppress_warning):
        LOGGER.warning("Thread '%s': missing ScriptRunContext", thread.name)
    return ctx
import streamlit