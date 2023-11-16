from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from airflow.callbacks.base_callback_sink import BaseCallbackSink
if TYPE_CHECKING:
    from multiprocessing.connection import Connection as MultiprocessingConnection
    from airflow.callbacks.callback_requests import CallbackRequest

class PipeCallbackSink(BaseCallbackSink):
    """
    Class for sending callbacks to DagProcessor using pipe.

    It is used when DagProcessor is not executed in standalone mode.
    """

    def __init__(self, get_sink_pipe: Callable[[], MultiprocessingConnection]):
        if False:
            i = 10
            return i + 15
        self._get_sink_pipe = get_sink_pipe

    def send(self, callback: CallbackRequest):
        if False:
            print('Hello World!')
        '\n        Send information about the callback to be executed by Pipe.\n\n        :param callback: Callback request to be executed.\n        '
        try:
            self._get_sink_pipe().send(callback)
        except ConnectionError:
            pass