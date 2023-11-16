from datetime import datetime
from typing import Optional, List, Dict
from uuid import uuid4
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from posthog.clickhouse.log_entries import INSERT_LOG_ENTRY_SQL
from posthog.kafka_client.client import ClickhouseProducer
from posthog.kafka_client.topics import KAFKA_CLICKHOUSE_SESSION_REPLAY_EVENTS, KAFKA_LOG_ENTRIES
from posthog.models.event.util import format_clickhouse_timestamp
from posthog.utils import cast_timestamp_or_now
INSERT_SINGLE_SESSION_REPLAY = "\nINSERT INTO sharded_session_replay_events (\n    session_id,\n    team_id,\n    distinct_id,\n    min_first_timestamp,\n    max_last_timestamp,\n    first_url,\n    click_count,\n    keypress_count,\n    mouse_activity_count,\n    active_milliseconds,\n    console_log_count,\n    console_warn_count,\n    console_error_count\n)\nSELECT\n    %(session_id)s,\n    %(team_id)s,\n    %(distinct_id)s,\n    toDateTime64(%(first_timestamp)s, 6, 'UTC'),\n    toDateTime64(%(last_timestamp)s, 6, 'UTC'),\n    argMinState(cast(%(first_url)s, 'Nullable(String)'), toDateTime64(%(first_timestamp)s, 6, 'UTC')),\n    %(click_count)s,\n    %(keypress_count)s,\n    %(mouse_activity_count)s,\n    %(active_milliseconds)s,\n    %(console_log_count)s,\n    %(console_warn_count)s,\n    %(console_error_count)s\n"

def _sensible_first_timestamp(first_timestamp: Optional[str | datetime], last_timestamp: Optional[str | datetime]) -> str:
    if False:
        while True:
            i = 10
    '\n    Normalise the first timestamp to be used in the session replay summary.\n    If it is not provided but there is a last_timestamp, use an hour before that last_timestamp\n    Otherwise we use the current time\n    '
    sensible_timestamp = None
    if first_timestamp is not None:
        if not isinstance(first_timestamp, str):
            sensible_timestamp = first_timestamp.isoformat()
        else:
            sensible_timestamp = first_timestamp
    elif last_timestamp is not None:
        if isinstance(last_timestamp, str):
            last_timestamp = parse(last_timestamp)
        sensible_timestamp = (last_timestamp - relativedelta(seconds=3600)).isoformat()
    return format_clickhouse_timestamp(cast_timestamp_or_now(sensible_timestamp))

def _sensible_last_timestamp(first_timestamp: Optional[str | datetime], last_timestamp: Optional[str | datetime]) -> str:
    if False:
        print('Hello World!')
    '\n    Normalise the last timestamp to be used in the session replay summary.\n    If it is not provided but there is a first_timestamp, use an hour after that last_timestamp\n    Otherwise we use the current time\n    '
    sensible_timestamp = None
    if last_timestamp is not None:
        if not isinstance(last_timestamp, str):
            sensible_timestamp = last_timestamp.isoformat()
        else:
            sensible_timestamp = last_timestamp
    elif first_timestamp is not None:
        if isinstance(first_timestamp, str):
            first_timestamp = parse(first_timestamp)
        sensible_timestamp = (first_timestamp - relativedelta(seconds=3600)).isoformat()
    return format_clickhouse_timestamp(cast_timestamp_or_now(sensible_timestamp))

def produce_replay_summary(team_id: int, session_id: Optional[str]=None, distinct_id: Optional[str]=None, first_timestamp: Optional[str | datetime]=None, last_timestamp: Optional[str | datetime]=None, first_url: Optional[str | None]=None, click_count: Optional[int]=None, keypress_count: Optional[int]=None, mouse_activity_count: Optional[int]=None, active_milliseconds: Optional[float]=None, console_log_count: Optional[int]=None, console_warn_count: Optional[int]=None, console_error_count: Optional[int]=None, log_messages: Dict[str, List[str]] | None=None):
    if False:
        i = 10
        return i + 15
    if log_messages is None:
        log_messages = {}
    first_timestamp = _sensible_first_timestamp(first_timestamp, last_timestamp)
    last_timestamp = _sensible_last_timestamp(first_timestamp, last_timestamp)
    timestamp = format_clickhouse_timestamp(cast_timestamp_or_now(first_timestamp))
    data = {'session_id': session_id or '1', 'team_id': team_id, 'distinct_id': distinct_id or 'user', 'first_timestamp': timestamp, 'last_timestamp': format_clickhouse_timestamp(cast_timestamp_or_now(last_timestamp)), 'first_url': first_url, 'click_count': click_count or 0, 'keypress_count': keypress_count or 0, 'mouse_activity_count': mouse_activity_count or 0, 'active_milliseconds': active_milliseconds or 0, 'console_log_count': console_log_count or 0, 'console_warn_count': console_warn_count or 0, 'console_error_count': console_error_count or 0}
    p = ClickhouseProducer()
    p.produce(topic=KAFKA_CLICKHOUSE_SESSION_REPLAY_EVENTS, sql=INSERT_SINGLE_SESSION_REPLAY, data=data)
    for (level, messages) in log_messages.items():
        for message in messages:
            p.produce(topic=KAFKA_LOG_ENTRIES, sql=INSERT_LOG_ENTRY_SQL, data={'team_id': team_id, 'message': message, 'level': level, 'log_source': 'session_replay', 'log_source_id': session_id, 'instance_id': str(uuid4()), 'timestamp': timestamp})