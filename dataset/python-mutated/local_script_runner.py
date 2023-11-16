from __future__ import annotations
import os
import time
from typing import Any
from urllib import parse
from streamlit import runtime
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_run_context import ScriptRunContext
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.testing.v1.element_tree import ElementTree, parse_tree_from_messages

class LocalScriptRunner(ScriptRunner):
    """Subclasses ScriptRunner to provide some testing features."""

    def __init__(self, script_path: str, session_state: SafeSessionState):
        if False:
            return 10
        'Initializes the ScriptRunner for the given script_path.'
        assert os.path.isfile(script_path), f'File not found at {script_path}'
        self.forward_msg_queue = ForwardMsgQueue()
        self.script_path = script_path
        self.session_state = session_state
        super().__init__(session_id='test session id', main_script_path=script_path, session_state=self.session_state._state, uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), script_cache=ScriptCache(), initial_rerun_data=RerunData(), user_info={'email': 'test@test.com'})
        self.events: list[ScriptRunnerEvent] = []
        self.event_data: list[Any] = []

        def record_event(sender: ScriptRunner | None, event: ScriptRunnerEvent, **kwargs) -> None:
            if False:
                print('Hello World!')
            assert sender is None or sender == self, 'Unexpected ScriptRunnerEvent sender!'
            self.events.append(event)
            self.event_data.append(kwargs)
            if event == ScriptRunnerEvent.ENQUEUE_FORWARD_MSG:
                forward_msg = kwargs['forward_msg']
                self.forward_msg_queue.enqueue(forward_msg)
        self.on_event.connect(record_event, weak=False)

    def join(self) -> None:
        if False:
            return 10
        'Wait for the script thread to finish, if it is running.'
        if self._script_thread is not None:
            self._script_thread.join()

    def forward_msgs(self) -> list[ForwardMsg]:
        if False:
            return 10
        'Return all messages in our ForwardMsgQueue.'
        return self.forward_msg_queue._queue

    def run(self, widget_state: WidgetStates | None=None, query_params=None, timeout: float=3) -> ElementTree:
        if False:
            return 10
        'Run the script, and parse the output messages for querying\n        and interaction.\n\n        Timeout is in seconds.\n        '
        query_string = ''
        if query_params:
            query_string = parse.urlencode(query_params, doseq=True)
        rerun_data = RerunData(widget_states=widget_state, query_string=query_string)
        self.request_rerun(rerun_data)
        if not self._script_thread:
            self.start()
        require_widgets_deltas(self, timeout)
        tree = parse_tree_from_messages(self.forward_msgs())
        return tree

    def script_stopped(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        for e in self.events:
            if e == ScriptRunnerEvent.SHUTDOWN:
                return True
        return False

    def _on_script_finished(self, ctx: ScriptRunContext, event: ScriptRunnerEvent, premature_stop: bool) -> None:
        if False:
            while True:
                i = 10
        if not premature_stop:
            self._session_state._state._remove_stale_widgets(ctx.widget_ids_this_run)
        self.on_event.send(self, event=event)
        runtime.get_instance().media_file_mgr.remove_orphaned_files()

def require_widgets_deltas(runner: LocalScriptRunner, timeout: float=3) -> None:
    if False:
        print('Hello World!')
    'Wait for the given ScriptRunner to emit a completion event. If the timeout\n    is reached, the runner will be shutdown and an error will be thrown.\n    '
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.001)
        if runner.script_stopped():
            return
    err_string = f'AppTest script run timed out after {timeout}s)'
    runner.request_stop()
    runner.join()
    raise RuntimeError(err_string)