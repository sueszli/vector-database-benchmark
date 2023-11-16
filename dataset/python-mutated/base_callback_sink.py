from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from airflow.callbacks.callback_requests import CallbackRequest

class BaseCallbackSink:
    """Base class for Callbacks Sinks."""

    def send(self, callback: CallbackRequest) -> None:
        if False:
            while True:
                i = 10
        'Send callback for execution.'
        raise NotImplementedError()