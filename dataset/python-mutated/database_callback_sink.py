from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.callbacks.base_callback_sink import BaseCallbackSink
from airflow.models.db_callback_request import DbCallbackRequest
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.callbacks.callback_requests import CallbackRequest

class DatabaseCallbackSink(BaseCallbackSink):
    """Sends callbacks to database."""

    @provide_session
    def send(self, callback: CallbackRequest, session: Session=NEW_SESSION) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Send callback for execution.'
        db_callback = DbCallbackRequest(callback=callback, priority_weight=10)
        session.add(db_callback)