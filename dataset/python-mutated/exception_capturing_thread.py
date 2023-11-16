import threading
from typing import Any, Callable, Optional
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import ScriptRunContext, add_script_run_ctx
from streamlit.runtime.state import SafeSessionState, SessionState

def call_on_threads(func: Callable[[int], Any], num_threads: int, timeout: Optional[float]=0.25, attach_script_run_ctx: bool=True) -> None:
    if False:
        while True:
            i = 10
    "Call a function on multiple threads simultaneously and assert that no\n    thread raises an unhandled exception.\n\n    The function must take single `int` param, which will be the index of\n    the thread it's being called on.\n\n    Note that a passing multi-threaded test does not generally guarantee that\n    the tested code is thread safe! Because threading issues tend to be\n    non-deterministic, a flaky test that fails only occasionally is a good\n    indicator of an underlying issue.\n\n    Parameters\n    ----------\n    func\n        The function to call on each thread.\n    num_threads\n        The number of threads to create.\n    timeout\n        If the thread runs for longer than this amount of time, raise an\n        Exception.\n    attach_script_run_ctx\n        If True, attach a mock ScriptRunContext to each thread before\n        starting.\n    "
    threads = [ExceptionCapturingThread(name=f'Thread {ii}', target=func, args=[ii]) for ii in range(num_threads)]
    if attach_script_run_ctx:
        for ii in range(num_threads):
            ctx = ScriptRunContext(session_id=f'Thread{ii}_Session', _enqueue=ForwardMsgQueue().enqueue, query_string='', session_state=SafeSessionState(SessionState(), lambda : None), uploaded_file_mgr=MemoryUploadedFileManager('/mock/upload'), page_script_hash='', user_info={'email': 'test@test.com'})
            thread = threads[ii]
            add_script_run_ctx(thread, ctx)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        thread.assert_no_unhandled_exception()

class ExceptionCapturingThread(threading.Thread):
    """Thread subclass that captures unhandled exceptions."""

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if False:
            print('Hello World!')
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self._unhandled_exception: Optional[BaseException] = None

    @property
    def unhandled_exception(self) -> Optional[BaseException]:
        if False:
            print('Hello World!')
        "The unhandled exception raised by the thread's target, if it raised one."
        return self._unhandled_exception

    def assert_no_unhandled_exception(self) -> None:
        if False:
            print('Hello World!')
        'If the thread target raised an unhandled exception, re-raise it.\n        Otherwise no-op.\n        '
        if self._unhandled_exception is not None:
            raise RuntimeError(f"Unhandled exception in thread '{self.name}'") from self._unhandled_exception

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            super().run()
        except Exception as e:
            self._unhandled_exception = e