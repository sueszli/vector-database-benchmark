from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.segment.hooks.segment import SegmentHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class SegmentTrackEventOperator(BaseOperator):
    """
    Send Track Event to Segment for a specified user_id and event.

    :param user_id: The ID for this user in your database. (templated)
    :param event: The name of the event you're tracking. (templated)
    :param properties: A dictionary of properties for the event. (templated)
    :param segment_conn_id: The connection ID to use when connecting to Segment.
    :param segment_debug_mode: Determines whether Segment should run in debug mode.
        Defaults to False
    """
    template_fields: Sequence[str] = ('user_id', 'event', 'properties')
    ui_color = '#ffd700'

    def __init__(self, *, user_id: str, event: str, properties: dict | None=None, segment_conn_id: str='segment_default', segment_debug_mode: bool=False, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.user_id = user_id
        self.event = event
        properties = properties or {}
        self.properties = properties
        self.segment_debug_mode = segment_debug_mode
        self.segment_conn_id = segment_conn_id

    def execute(self, context: Context) -> None:
        if False:
            i = 10
            return i + 15
        hook = SegmentHook(segment_conn_id=self.segment_conn_id, segment_debug_mode=self.segment_debug_mode)
        self.log.info('Sending track event (%s) for user id: %s with properties: %s', self.event, self.user_id, self.properties)
        hook.track(user_id=self.user_id, event=self.event, properties=self.properties)