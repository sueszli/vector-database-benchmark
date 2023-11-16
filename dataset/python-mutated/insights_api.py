from datetime import datetime, timedelta
from math import ceil
from time import sleep
from typing import Optional, Tuple, Union
import zoneinfo
from rest_framework import request
from posthog.caching.calculate_results import CLICKHOUSE_MAX_EXECUTION_TIME, calculate_cache_key
from posthog.caching.insight_caching_state import InsightCachingState
from posthog.models import DashboardTile, Insight
from posthog.models.filters.utils import get_filter
from posthog.utils import refresh_requested_by_client
'\nUtilities used by the insights API to determine whether\nor not to refresh an insight upon a client request to do so\n'
BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL = timedelta(minutes=15)
REDUCED_MINIMUM_INSIGHT_REFRESH_INTERVAL = timedelta(minutes=3)
INCREASED_MINIMUM_INSIGHT_REFRESH_INTERVAL = timedelta(minutes=30)

def should_refresh_insight(insight: Insight, dashboard_tile: Optional[DashboardTile], *, request: request.Request, is_shared=False) -> Tuple[bool, timedelta]:
    if False:
        while True:
            i = 10
    "Return whether the insight should be refreshed now, and what's the minimum wait time between refreshes.\n\n    If a refresh already is being processed somewhere else, this function will wait for that to finish (or time out).\n    "
    filter = get_filter(data=insight.dashboard_filters(dashboard_tile.dashboard if dashboard_tile is not None else None), team=insight.team)
    delta_days: Optional[int] = None
    if filter.date_from and filter.date_to:
        delta = filter.date_to - filter.date_from
        delta_days = ceil(delta.total_seconds() / timedelta(days=1).total_seconds())
    refresh_frequency = BASE_MINIMUM_INSIGHT_REFRESH_INTERVAL
    if is_shared:
        refresh_frequency = INCREASED_MINIMUM_INSIGHT_REFRESH_INTERVAL
    elif getattr(filter, 'interval', None) == 'hour' or (delta_days is not None and delta_days <= 7):
        refresh_frequency = REDUCED_MINIMUM_INSIGHT_REFRESH_INTERVAL
    refresh_insight_now = False
    if refresh_requested_by_client(request):
        now = datetime.now(tz=zoneinfo.ZoneInfo('UTC'))
        target: Union[Insight, DashboardTile] = insight if dashboard_tile is None else dashboard_tile
        cache_key = calculate_cache_key(target)
        caching_state = InsightCachingState.objects.filter(team_id=insight.team.pk, cache_key=cache_key, insight=insight).order_by('-last_refresh_queued_at').first()
        refresh_insight_now = caching_state is None or caching_state.last_refresh is None or caching_state.last_refresh + refresh_frequency <= now
        if refresh_insight_now:
            has_refreshed_somewhere_else = _sleep_if_refresh_is_running_somewhere_else(caching_state, now)
            if has_refreshed_somewhere_else:
                refresh_insight_now = False
    return (refresh_insight_now, refresh_frequency)

def _sleep_if_refresh_is_running_somewhere_else(caching_state: Optional[InsightCachingState], now: datetime) -> bool:
    if False:
        i = 10
        return i + 15
    'Prevent the same query from running concurrently needlessly.'
    is_refresh_currently_running = _is_refresh_currently_running_somewhere_else(caching_state, now)
    if is_refresh_currently_running:
        assert caching_state is not None
        while is_refresh_currently_running:
            sleep(1)
            caching_state.refresh_from_db()
            has_refresh_completed = caching_state.last_refresh is not None and caching_state.last_refresh >= caching_state.last_refresh_queued_at
            if has_refresh_completed:
                return True
            is_refresh_currently_running = _is_refresh_currently_running_somewhere_else(caching_state, datetime.now(tz=zoneinfo.ZoneInfo('UTC')))
    return False

def _is_refresh_currently_running_somewhere_else(caching_state: Optional[InsightCachingState], now: datetime) -> bool:
    if False:
        print('Hello World!')
    'Return whether the refresh is most likely still running somewhere else.'
    if caching_state is not None and caching_state.last_refresh_queued_at is not None and (caching_state.last_refresh_queued_at > now - timedelta(seconds=CLICKHOUSE_MAX_EXECUTION_TIME)) and (caching_state.last_refresh is None or caching_state.last_refresh < caching_state.last_refresh_queued_at):
        return True
    else:
        return False