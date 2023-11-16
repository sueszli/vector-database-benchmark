from datetime import datetime, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo
from dateutil.parser import isoparse
from freezegun import freeze_time
from posthog.clickhouse.client import sync_execute
from posthog.models import Team
from posthog.models.event.util import format_clickhouse_timestamp
from posthog.queries.app_metrics.serializers import AppMetricsRequestSerializer
from posthog.session_recordings.queries.test.session_replay_sql import produce_replay_summary
from posthog.test.base import BaseTest, ClickhouseTestMixin, snapshot_clickhouse_queries

def make_filter(serializer_klass=AppMetricsRequestSerializer, **kwargs) -> AppMetricsRequestSerializer:
    if False:
        while True:
            i = 10
    filter = serializer_klass(data=kwargs)
    filter.is_valid(raise_exception=True)
    return filter

class SessionReplaySummaryQuery:

    def __init__(self, team: Team, session_id: str, reference_date: str):
        if False:
            while True:
                i = 10
        self.team = team
        self.session_id = session_id
        self.reference_date = reference_date

    def list_all(self):
        if False:
            return 10
        params = {'team_id': self.team.pk, 'start_time': format_clickhouse_timestamp(isoparse(self.reference_date) - timedelta(hours=48)), 'end_time': format_clickhouse_timestamp(isoparse(self.reference_date) + timedelta(hours=48)), 'session_ids': (self.session_id,)}
        results = sync_execute("\n            select\n               session_id,\n               any(team_id),\n               any(distinct_id),\n               min(min_first_timestamp),\n               max(max_last_timestamp),\n               dateDiff('SECOND', min(min_first_timestamp), max(max_last_timestamp)) as duration,\n               argMinMerge(first_url) as first_url,\n               sum(click_count),\n               sum(keypress_count),\n               sum(mouse_activity_count),\n               round((sum(active_milliseconds)/1000)/duration, 2) as active_time\n            from session_replay_events\n            prewhere team_id = %(team_id)s\n            and min_first_timestamp >= %(start_time)s\n            and max_last_timestamp <= %(end_time)s\n            and session_id in %(session_ids)s\n            group by session_id\n            ", params)
        return results

class TestReceiveSummarizedSessionReplays(ClickhouseTestMixin, BaseTest):

    @snapshot_clickhouse_queries
    @freeze_time('2023-01-04T12:34')
    def test_session_replay_summaries_can_be_queried(self):
        if False:
            print('Hello World!')
        session_id = 'test_session_replay_summaries_can_be_queried-session-id'
        produce_replay_summary(session_id=session_id, team_id=self.team.pk, first_timestamp='2023-04-27 10:00:00.309', last_timestamp='2023-04-27 14:20:42.237', distinct_id=str(self.user.distinct_id), first_url='https://first-url-ingested.com', click_count=2, keypress_count=2, mouse_activity_count=2, active_milliseconds=33624 * 1000 * 0.3)
        produce_replay_summary(session_id=session_id, team_id=self.team.pk, first_timestamp='2023-04-27T19:17:38.116', last_timestamp='2023-04-27T19:17:38.117', distinct_id=str(self.user.distinct_id), first_url='https://second-url-ingested.com', click_count=2, keypress_count=2, mouse_activity_count=2)
        produce_replay_summary(session_id=session_id, team_id=self.team.pk, first_timestamp='2023-04-27T19:18:24.597', last_timestamp='2023-04-27T19:20:24.597', distinct_id=str(self.user.distinct_id), first_url='https://third-url-ingested.com', click_count=2, keypress_count=2, mouse_activity_count=2)
        produce_replay_summary(session_id=session_id, team_id=self.team.pk, first_timestamp='2023-04-26T19:18:24.597', last_timestamp='2023-04-29T19:20:24.597', distinct_id=str(self.user.distinct_id), first_url=None, click_count=2, keypress_count=2, mouse_activity_count=2)
        produce_replay_summary(session_id=session_id, team_id=self.team.pk + 100, first_timestamp='2023-04-26T19:18:24.597', last_timestamp='2023-04-28T19:20:24.597', distinct_id=str(self.user.distinct_id), first_url=None, click_count=2, keypress_count=2, mouse_activity_count=2)
        produce_replay_summary(session_id=str(uuid4()), team_id=self.team.pk, first_timestamp='2023-04-26T19:18:24.597', last_timestamp='2023-04-26T19:20:24.597', distinct_id=str(self.user.distinct_id), first_url=None, click_count=2, keypress_count=2, mouse_activity_count=2)
        results = SessionReplaySummaryQuery(self.team, session_id, '2023-04-26T19:18:24.597').list_all()
        assert results == [(session_id, self.team.pk, str(self.user.distinct_id), datetime(2023, 4, 27, 10, 0, 0, 309000, tzinfo=ZoneInfo('UTC')), datetime(2023, 4, 27, 19, 20, 24, 597000, tzinfo=ZoneInfo('UTC')), 33624, 'https://first-url-ingested.com', 6, 6, 6, 0.3)]