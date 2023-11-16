import dataclasses
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, cast
import requests
from retry import retry
import structlog
from dateutil import parser
from django.conf import settings
from django.db import connection
from django.db.models import Count, Q
from posthoganalytics.client import Client
from psycopg2 import sql
from sentry_sdk import capture_exception
from posthog import version_requirement
from posthog.celery import app
from posthog.clickhouse.client.connection import Workload
from posthog.client import sync_execute
from posthog.cloud_utils import get_cached_instance_license, is_cloud
from posthog.constants import FlagRequestType
from posthog.logging.timing import timed_log
from posthog.models import GroupTypeMapping, OrganizationMembership, User
from posthog.models.dashboard import Dashboard
from posthog.models.feature_flag import FeatureFlag
from posthog.models.organization import Organization
from posthog.models.plugin import PluginConfig
from posthog.models.team.team import Team
from posthog.models.utils import namedtuplefetchall
from posthog.settings import CLICKHOUSE_CLUSTER, INSTANCE_TAG
from posthog.utils import get_helm_info_env, get_instance_realm, get_instance_region, get_machine_id, get_previous_day
logger = structlog.get_logger(__name__)
Period = TypedDict('Period', {'start_inclusive': str, 'end_inclusive': str})
TableSizes = TypedDict('TableSizes', {'posthog_event': int, 'posthog_sessionrecordingevent': int})
CH_BILLING_SETTINGS = {'max_execution_time': 5 * 60}
QUERY_RETRIES = 3
QUERY_RETRY_DELAY = 1
QUERY_RETRY_BACKOFF = 2

@dataclasses.dataclass
class UsageReportCounters:
    event_count_lifetime: int
    event_count_in_period: int
    event_count_in_month: int
    event_count_with_groups_in_period: int
    recording_count_in_period: int
    recording_count_total: int
    group_types_total: int
    dashboard_count: int
    dashboard_template_count: int
    dashboard_shared_count: int
    dashboard_tagged_count: int
    ff_count: int
    ff_active_count: int
    decide_requests_count_in_period: int
    decide_requests_count_in_month: int
    local_evaluation_requests_count_in_period: int
    local_evaluation_requests_count_in_month: int
    billable_feature_flag_requests_count_in_period: int
    billable_feature_flag_requests_count_in_month: int
    hogql_app_bytes_read: int
    hogql_app_rows_read: int
    hogql_app_duration_ms: int
    hogql_api_bytes_read: int
    hogql_api_rows_read: int
    hogql_api_duration_ms: int
    event_explorer_app_bytes_read: int
    event_explorer_app_rows_read: int
    event_explorer_app_duration_ms: int
    event_explorer_api_bytes_read: int
    event_explorer_api_rows_read: int
    event_explorer_api_duration_ms: int
    survey_responses_count_in_period: int
    survey_responses_count_in_month: int

@dataclasses.dataclass
class InstanceMetadata:
    deployment_infrastructure: str
    realm: str
    period: Period
    site_url: str
    product: str
    helm: Optional[dict]
    clickhouse_version: Optional[str]
    users_who_logged_in: Optional[List[Dict[str, Union[str, int]]]]
    users_who_logged_in_count: Optional[int]
    users_who_signed_up: Optional[List[Dict[str, Union[str, int]]]]
    users_who_signed_up_count: Optional[int]
    table_sizes: Optional[TableSizes]
    plugins_installed: Optional[Dict]
    plugins_enabled: Optional[Dict]
    instance_tag: str

@dataclasses.dataclass
class OrgReport(UsageReportCounters):
    date: str
    organization_id: str
    organization_name: str
    organization_created_at: str
    organization_user_count: int
    team_count: int
    teams: Dict[str, UsageReportCounters]

@dataclasses.dataclass
class FullUsageReport(OrgReport, InstanceMetadata):
    pass

def fetch_table_size(table_name: str) -> int:
    if False:
        return 10
    return fetch_sql('SELECT pg_total_relation_size(%s) as size', (table_name,))[0].size

def fetch_sql(sql_: str, params: Tuple[Any, ...]) -> List[Any]:
    if False:
        i = 10
        return i + 15
    with connection.cursor() as cursor:
        cursor.execute(sql.SQL(sql_), params)
        return namedtuplefetchall(cursor)

def get_product_name(realm: str, has_license: bool) -> str:
    if False:
        i = 10
        return i + 15
    if realm == 'cloud':
        return 'cloud'
    elif realm in {'hosted', 'hosted-clickhouse'}:
        return 'scale' if has_license else 'open source'
    else:
        return 'unknown'

def get_instance_metadata(period: Tuple[datetime, datetime]) -> InstanceMetadata:
    if False:
        return 10
    has_license = False
    if settings.EE_AVAILABLE:
        license = get_cached_instance_license()
        has_license = license is not None
    (period_start, period_end) = period
    realm = get_instance_realm()
    metadata = InstanceMetadata(deployment_infrastructure=os.getenv('DEPLOYMENT', 'unknown'), realm=realm, period={'start_inclusive': period_start.isoformat(), 'end_inclusive': period_end.isoformat()}, site_url=settings.SITE_URL, product=get_product_name(realm, has_license), helm=None, clickhouse_version=None, users_who_logged_in=None, users_who_logged_in_count=None, users_who_signed_up=None, users_who_signed_up_count=None, table_sizes=None, plugins_installed=None, plugins_enabled=None, instance_tag=INSTANCE_TAG)
    if realm != 'cloud':
        metadata.helm = get_helm_info_env()
        metadata.clickhouse_version = str(version_requirement.get_clickhouse_version())
        metadata.users_who_logged_in = [{'id': user.id, 'distinct_id': user.distinct_id} if user.anonymize_data else {'id': user.id, 'distinct_id': user.distinct_id, 'first_name': user.first_name, 'email': user.email} for user in User.objects.filter(is_active=True, last_login__gte=period_start, last_login__lte=period_end)]
        metadata.users_who_logged_in_count = len(metadata.users_who_logged_in)
        metadata.users_who_signed_up = [{'id': user.id, 'distinct_id': user.distinct_id} if user.anonymize_data else {'id': user.id, 'distinct_id': user.distinct_id, 'first_name': user.first_name, 'email': user.email} for user in User.objects.filter(is_active=True, date_joined__gte=period_start, date_joined__lte=period_end)]
        metadata.users_who_signed_up_count = len(metadata.users_who_signed_up)
        metadata.table_sizes = {'posthog_event': fetch_table_size('posthog_event'), 'posthog_sessionrecordingevent': fetch_table_size('posthog_sessionrecordingevent')}
        plugin_configs = PluginConfig.objects.select_related('plugin').all()
        metadata.plugins_installed = dict(Counter((plugin_config.plugin.name for plugin_config in plugin_configs)))
        metadata.plugins_enabled = dict(Counter((plugin_config.plugin.name for plugin_config in plugin_configs if plugin_config.enabled)))
    return metadata

def get_org_user_count(organization_id: str) -> int:
    if False:
        while True:
            i = 10
    return OrganizationMembership.objects.filter(organization_id=organization_id).count()

def get_org_owner_or_first_user(organization_id: str) -> Optional[User]:
    if False:
        i = 10
        return i + 15
    user = None
    membership = OrganizationMembership.objects.filter(organization_id=organization_id, level=OrganizationMembership.Level.OWNER).first()
    if not membership:
        membership = OrganizationMembership.objects.filter(organization_id=organization_id).first()
    if hasattr(membership, 'user'):
        membership = cast(OrganizationMembership, membership)
        user = membership.user
    else:
        capture_exception(Exception('No user found for org while generating report'), {'org': {'organization_id': organization_id}})
    return user

@app.task(ignore_result=True, autoretry_for=(Exception,), max_retries=3)
def send_report_to_billing_service(org_id: str, report: Dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not settings.EE_AVAILABLE:
        return
    from ee.billing.billing_manager import BillingManager, build_billing_token
    from ee.billing.billing_types import BillingStatus
    from ee.settings import BILLING_SERVICE_URL
    try:
        license = get_cached_instance_license()
        if not license or not license.is_v2_license:
            return
        organization = Organization.objects.get(id=org_id)
        if not organization:
            return
        token = build_billing_token(license, organization)
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        response = requests.post(f'{BILLING_SERVICE_URL}/api/usage', json=report, headers=headers)
        if response.status_code != 200:
            raise Exception(f'Failed to send usage report to billing service code:{response.status_code} response:{response.text}')
        logger.info(f'UsageReport sent to Billing for organization: {organization.id}')
        response_data: BillingStatus = response.json()
        BillingManager(license).update_org_details(organization, response_data)
        BillingManager(license).update_billing_distinct_ids(organization)
    except Exception as err:
        logger.error(f'UsageReport failed sending to Billing for organization: {organization.id}: {err}')
        capture_exception(err)
        pha_client = Client('sTMFPsFhdP1Ssg')
        capture_event(pha_client, f'organization usage report to billing service failure', org_id, {'err': str(err)})
        raise err

def capture_event(pha_client: Client, name: str, organization_id: str, properties: Dict[str, Any], timestamp: Optional[Union[datetime, str]]=None) -> None:
    if False:
        return 10
    if timestamp and isinstance(timestamp, str):
        try:
            timestamp = parser.isoparse(timestamp)
        except ValueError:
            timestamp = None
    if is_cloud():
        org_owner = get_org_owner_or_first_user(organization_id)
        distinct_id = org_owner.distinct_id if org_owner and org_owner.distinct_id else f'org-{organization_id}'
        pha_client.capture(distinct_id, name, {**properties, 'scope': 'user'}, groups={'organization': organization_id, 'instance': settings.SITE_URL}, timestamp=timestamp)
        pha_client.group_identify('organization', organization_id, properties)
    else:
        pha_client.capture(get_machine_id(), name, {**properties, 'scope': 'machine'}, groups={'instance': settings.SITE_URL}, timestamp=timestamp)
        pha_client.group_identify('instance', settings.SITE_URL, properties)

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_event_count_lifetime() -> List[Tuple[int, int]]:
    if False:
        while True:
            i = 10
    result = sync_execute('\n        SELECT team_id, count(1) as count\n        FROM events\n        GROUP BY team_id\n    ', workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_billable_event_count_in_period(begin: datetime, end: datetime, count_distinct: bool=False) -> List[Tuple[int, int]]:
    if False:
        i = 10
        return i + 15
    if count_distinct:
        distinct_expression = 'distinct toDate(timestamp), event, cityHash64(distinct_id), cityHash64(uuid)'
    else:
        distinct_expression = '1'
    result = sync_execute(f"\n        SELECT team_id, count({distinct_expression}) as count\n        FROM events\n        WHERE timestamp between %(begin)s AND %(end)s AND event != '$feature_flag_called'\n        GROUP BY team_id\n    ", {'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_event_count_with_groups_in_period(begin: datetime, end: datetime) -> List[Tuple[int, int]]:
    if False:
        while True:
            i = 10
    result = sync_execute("\n        SELECT team_id, count(1) as count\n        FROM events\n        WHERE timestamp between %(begin)s AND %(end)s\n        AND ($group_0 != '' OR $group_1 != '' OR $group_2 != '' OR $group_3 != '' OR $group_4 != '')\n        GROUP BY team_id\n    ", {'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_event_count_by_lib(begin: datetime, end: datetime) -> List[Tuple[int, str, int]]:
    if False:
        print('Hello World!')
    results = sync_execute("\n        SELECT team_id, JSONExtractString(properties, '$lib') as lib, COUNT(1) as count\n        FROM events\n        WHERE timestamp between %(begin)s AND %(end)s\n        GROUP BY lib, team_id\n    ", {'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return results

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_event_count_by_name(begin: datetime, end: datetime) -> List[Tuple[int, str, int]]:
    if False:
        i = 10
        return i + 15
    results = sync_execute('\n        SELECT team_id, event, COUNT(1) as count\n        FROM events\n        WHERE timestamp between %(begin)s AND %(end)s\n        GROUP BY event, team_id\n    ', {'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return results

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_recording_count_in_period(begin: datetime, end: datetime) -> List[Tuple[int, int]]:
    if False:
        for i in range(10):
            print('nop')
    previous_begin = begin - (end - begin)
    result = sync_execute('\n        SELECT team_id, count(distinct session_id) as count\n        FROM session_replay_events\n        WHERE min_first_timestamp BETWEEN %(begin)s AND %(end)s\n        AND session_id NOT IN (\n            -- we want to exclude sessions that might have events with timestamps\n            -- before the period we are interested in\n            SELECT DISTINCT session_id\n            FROM session_replay_events\n            -- begin is the very first instant of the period we are interested in\n            -- we assume it is also the very first instant of a day\n            -- so we can to subtract 1 second to get the day before\n            WHERE min_first_timestamp BETWEEN %(previous_begin)s AND %(begin)s\n            GROUP BY session_id\n        )\n        GROUP BY team_id\n    ', {'previous_begin': previous_begin, 'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_recording_count_total() -> List[Tuple[int, int]]:
    if False:
        return 10
    result = sync_execute('\n        SELECT team_id, count(distinct session_id) as count\n        FROM session_replay_events\n        GROUP BY team_id\n    ', workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_hogql_metric(begin: datetime, end: datetime, query_types: List[str], access_method: str='', metric: Literal['read_bytes', 'read_rows', 'query_duration_ms']='read_bytes') -> List[Tuple[int, int]]:
    if False:
        while True:
            i = 10
    if metric not in ['read_bytes', 'read_rows', 'query_duration_ms']:
        raise ValueError(f'Invalid metric {metric}')
    result = sync_execute(f"\n        WITH JSONExtractInt(log_comment, 'team_id') as team_id,\n             JSONExtractString(log_comment, 'query_type') as query_type,\n             JSONExtractString(log_comment, 'access_method') as access_method\n        SELECT team_id, sum({metric}) as count\n        FROM clusterAllReplicas({CLICKHOUSE_CLUSTER}, system.query_log)\n        WHERE (type = 'QueryFinish' OR type = 'ExceptionWhileProcessing')\n          AND is_initial_query = 1\n          AND query_type IN (%(query_types)s)\n          AND query_start_time between %(begin)s AND %(end)s\n          AND access_method = %(access_method)s\n        GROUP BY team_id\n    ", {'begin': begin, 'end': end, 'query_types': query_types, 'access_method': access_method}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_feature_flag_requests_count_in_period(begin: datetime, end: datetime, request_type: FlagRequestType) -> List[Tuple[int, int]]:
    if False:
        for i in range(10):
            print('nop')
    team_to_query = 1 if get_instance_region() == 'EU' else 2
    validity_token = settings.DECIDE_BILLING_ANALYTICS_TOKEN
    target_event = 'decide usage' if request_type == FlagRequestType.DECIDE else 'local evaluation usage'
    result = sync_execute('\n        SELECT distinct_id as team, sum(JSONExtractInt(properties, \'count\')) as sum\n        FROM events\n        WHERE team_id = %(team_to_query)s AND event=%(target_event)s AND timestamp between %(begin)s AND %(end)s\n        AND has([%(validity_token)s], replaceRegexpAll(JSONExtractRaw(properties, \'token\'), \'^"|"$\', \'\'))\n        GROUP BY team\n    ', {'begin': begin, 'end': end, 'team_to_query': team_to_query, 'validity_token': validity_token, 'target_event': target_event}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return result

@timed_log()
@retry(tries=QUERY_RETRIES, delay=QUERY_RETRY_DELAY, backoff=QUERY_RETRY_BACKOFF)
def get_teams_with_survey_responses_count_in_period(begin: datetime, end: datetime) -> List[Tuple[int, int]]:
    if False:
        i = 10
        return i + 15
    results = sync_execute("\n        SELECT team_id, COUNT() as count\n        FROM events\n        WHERE event = 'survey sent' AND timestamp between %(begin)s AND %(end)s\n        GROUP BY team_id\n    ", {'begin': begin, 'end': end}, workload=Workload.OFFLINE, settings=CH_BILLING_SETTINGS)
    return results

@app.task(ignore_result=True, max_retries=0)
def capture_report(capture_event_name: str, org_id: str, full_report_dict: Dict[str, Any], at_date: Optional[datetime]=None) -> None:
    if False:
        return 10
    pha_client = Client('sTMFPsFhdP1Ssg')
    try:
        capture_event(pha_client, capture_event_name, org_id, full_report_dict, timestamp=at_date)
        logger.info(f'UsageReport sent to PostHog for organization {org_id}')
    except Exception as err:
        logger.error(f'UsageReport sent to PostHog for organization {org_id} failed: {str(err)}')
        capture_event(pha_client, f'{capture_event_name} failure', org_id, {'error': str(err)})
    pha_client.flush()

def has_non_zero_usage(report: FullUsageReport) -> bool:
    if False:
        return 10
    return report.event_count_in_period > 0 or report.recording_count_in_period > 0 or report.decide_requests_count_in_period > 0 or (report.local_evaluation_requests_count_in_period > 0) or (report.survey_responses_count_in_period > 0)

def convert_team_usage_rows_to_dict(rows: List[Union[dict, Tuple[int, int]]]) -> Dict[int, int]:
    if False:
        for i in range(10):
            print('nop')
    team_id_map = {}
    for row in rows:
        if isinstance(row, dict) and 'team_id' in row:
            team_id_map[row['team_id']] = row['total']
        else:
            team_id_map[int(row[0])] = row[1]
    return team_id_map

def _get_all_usage_data(period_start: datetime, period_end: datetime) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets all usage data for the specified period. Clickhouse is good at counting things so\n    we count across all teams rather than doing it one by one\n    '
    return dict(teams_with_event_count_lifetime=get_teams_with_event_count_lifetime(), teams_with_event_count_in_period=get_teams_with_billable_event_count_in_period(period_start, period_end, count_distinct=True), teams_with_event_count_in_month=get_teams_with_billable_event_count_in_period(period_start.replace(day=1), period_end), teams_with_event_count_with_groups_in_period=get_teams_with_event_count_with_groups_in_period(period_start, period_end), teams_with_recording_count_in_period=get_teams_with_recording_count_in_period(period_start, period_end), teams_with_recording_count_total=get_teams_with_recording_count_total(), teams_with_decide_requests_count_in_period=get_teams_with_feature_flag_requests_count_in_period(period_start, period_end, FlagRequestType.DECIDE), teams_with_decide_requests_count_in_month=get_teams_with_feature_flag_requests_count_in_period(period_start.replace(day=1), period_end, FlagRequestType.DECIDE), teams_with_local_evaluation_requests_count_in_period=get_teams_with_feature_flag_requests_count_in_period(period_start, period_end, FlagRequestType.LOCAL_EVALUATION), teams_with_local_evaluation_requests_count_in_month=get_teams_with_feature_flag_requests_count_in_period(period_start.replace(day=1), period_end, FlagRequestType.LOCAL_EVALUATION), teams_with_group_types_total=list(GroupTypeMapping.objects.values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_dashboard_count=list(Dashboard.objects.values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_dashboard_template_count=list(Dashboard.objects.filter(creation_mode='template').values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_dashboard_shared_count=list(Dashboard.objects.filter(sharingconfiguration__enabled=True).values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_dashboard_tagged_count=list(Dashboard.objects.filter(tagged_items__isnull=False).values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_ff_count=list(FeatureFlag.objects.values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_ff_active_count=list(FeatureFlag.objects.filter(active=True).values('team_id').annotate(total=Count('id')).order_by('team_id')), teams_with_hogql_app_bytes_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_bytes', query_types=['hogql_query', 'HogQLQuery'], access_method=''), teams_with_hogql_app_rows_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_rows', query_types=['hogql_query', 'HogQLQuery'], access_method=''), teams_with_hogql_app_duration_ms=get_teams_with_hogql_metric(period_start, period_end, metric='query_duration_ms', query_types=['hogql_query', 'HogQLQuery'], access_method=''), teams_with_hogql_api_bytes_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_bytes', query_types=['hogql_query', 'HogQLQuery'], access_method='personal_api_key'), teams_with_hogql_api_rows_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_rows', query_types=['hogql_query', 'HogQLQuery'], access_method='personal_api_key'), teams_with_hogql_api_duration_ms=get_teams_with_hogql_metric(period_start, period_end, metric='query_duration_ms', query_types=['hogql_query', 'HogQLQuery'], access_method='personal_api_key'), teams_with_event_explorer_app_bytes_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_bytes', query_types=['EventsQuery'], access_method=''), teams_with_event_explorer_app_rows_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_rows', query_types=['EventsQuery'], access_method=''), teams_with_event_explorer_app_duration_ms=get_teams_with_hogql_metric(period_start, period_end, metric='query_duration_ms', query_types=['EventsQuery'], access_method=''), teams_with_event_explorer_api_bytes_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_bytes', query_types=['EventsQuery'], access_method='personal_api_key'), teams_with_event_explorer_api_rows_read=get_teams_with_hogql_metric(period_start, period_end, metric='read_rows', query_types=['EventsQuery'], access_method='personal_api_key'), teams_with_event_explorer_api_duration_ms=get_teams_with_hogql_metric(period_start, period_end, metric='query_duration_ms', query_types=['EventsQuery'], access_method='personal_api_key'), teams_with_survey_responses_count_in_period=get_teams_with_survey_responses_count_in_period(period_start, period_end), teams_with_survey_responses_count_in_month=get_teams_with_survey_responses_count_in_period(period_start.replace(day=1), period_end))

def _get_all_usage_data_as_team_rows(period_start: datetime, period_end: datetime) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Gets all usage data for the specified period as a map of team_id -> value. This makes it faster\n    to access the data than looping over all_data to find what we want.\n    '
    all_data = _get_all_usage_data(period_start, period_end)
    for (key, rows) in all_data.items():
        all_data[key] = convert_team_usage_rows_to_dict(rows)
    return all_data

def _get_teams_for_usage_reports() -> Sequence[Team]:
    if False:
        print('Hello World!')
    return list(Team.objects.select_related('organization').exclude(Q(organization__for_internal_metrics=True) | Q(is_demo=True)))

def _get_team_report(all_data: Dict[str, Any], team: Team) -> UsageReportCounters:
    if False:
        print('Hello World!')
    decide_requests_count_in_month = all_data['teams_with_decide_requests_count_in_month'].get(team.id, 0)
    decide_requests_count_in_period = all_data['teams_with_decide_requests_count_in_period'].get(team.id, 0)
    local_evaluation_requests_count_in_period = all_data['teams_with_local_evaluation_requests_count_in_period'].get(team.id, 0)
    local_evaluation_requests_count_in_month = all_data['teams_with_local_evaluation_requests_count_in_month'].get(team.id, 0)
    return UsageReportCounters(event_count_lifetime=all_data['teams_with_event_count_lifetime'].get(team.id, 0), event_count_in_period=all_data['teams_with_event_count_in_period'].get(team.id, 0), event_count_in_month=all_data['teams_with_event_count_in_month'].get(team.id, 0), event_count_with_groups_in_period=all_data['teams_with_event_count_with_groups_in_period'].get(team.id, 0), recording_count_in_period=all_data['teams_with_recording_count_in_period'].get(team.id, 0), recording_count_total=all_data['teams_with_recording_count_total'].get(team.id, 0), group_types_total=all_data['teams_with_group_types_total'].get(team.id, 0), decide_requests_count_in_period=decide_requests_count_in_period, decide_requests_count_in_month=decide_requests_count_in_month, local_evaluation_requests_count_in_period=local_evaluation_requests_count_in_period, local_evaluation_requests_count_in_month=local_evaluation_requests_count_in_month, billable_feature_flag_requests_count_in_month=decide_requests_count_in_month + local_evaluation_requests_count_in_month * 10, billable_feature_flag_requests_count_in_period=decide_requests_count_in_period + local_evaluation_requests_count_in_period * 10, dashboard_count=all_data['teams_with_dashboard_count'].get(team.id, 0), dashboard_template_count=all_data['teams_with_dashboard_template_count'].get(team.id, 0), dashboard_shared_count=all_data['teams_with_dashboard_shared_count'].get(team.id, 0), dashboard_tagged_count=all_data['teams_with_dashboard_tagged_count'].get(team.id, 0), ff_count=all_data['teams_with_ff_count'].get(team.id, 0), ff_active_count=all_data['teams_with_ff_active_count'].get(team.id, 0), hogql_app_bytes_read=all_data['teams_with_hogql_app_bytes_read'].get(team.id, 0), hogql_app_rows_read=all_data['teams_with_hogql_app_rows_read'].get(team.id, 0), hogql_app_duration_ms=all_data['teams_with_hogql_app_duration_ms'].get(team.id, 0), hogql_api_bytes_read=all_data['teams_with_hogql_api_bytes_read'].get(team.id, 0), hogql_api_rows_read=all_data['teams_with_hogql_api_rows_read'].get(team.id, 0), hogql_api_duration_ms=all_data['teams_with_hogql_api_duration_ms'].get(team.id, 0), event_explorer_app_bytes_read=all_data['teams_with_event_explorer_app_bytes_read'].get(team.id, 0), event_explorer_app_rows_read=all_data['teams_with_event_explorer_app_rows_read'].get(team.id, 0), event_explorer_app_duration_ms=all_data['teams_with_event_explorer_app_duration_ms'].get(team.id, 0), event_explorer_api_bytes_read=all_data['teams_with_event_explorer_api_bytes_read'].get(team.id, 0), event_explorer_api_rows_read=all_data['teams_with_event_explorer_api_rows_read'].get(team.id, 0), event_explorer_api_duration_ms=all_data['teams_with_event_explorer_api_duration_ms'].get(team.id, 0), survey_responses_count_in_period=all_data['teams_with_survey_responses_count_in_period'].get(team.id, 0), survey_responses_count_in_month=all_data['teams_with_survey_responses_count_in_month'].get(team.id, 0))

def _add_team_report_to_org_reports(org_reports: Dict[str, OrgReport], team: Team, team_report: UsageReportCounters, period_start: datetime) -> None:
    if False:
        return 10
    org_id = str(team.organization.id)
    if org_id not in org_reports:
        org_report = OrgReport(date=period_start.strftime('%Y-%m-%d'), organization_id=org_id, organization_name=team.organization.name, organization_created_at=team.organization.created_at.isoformat(), organization_user_count=get_org_user_count(org_id), team_count=1, teams={str(team.id): team_report}, **dataclasses.asdict(team_report))
        org_reports[org_id] = org_report
    else:
        org_report = org_reports[org_id]
        org_report.teams[str(team.id)] = team_report
        org_report.team_count += 1
        for field in dataclasses.fields(UsageReportCounters):
            if hasattr(team_report, field.name):
                setattr(org_report, field.name, getattr(org_report, field.name) + getattr(team_report, field.name))

def _get_all_org_reports(period_start: datetime, period_end: datetime) -> Dict[str, OrgReport]:
    if False:
        i = 10
        return i + 15
    all_data = _get_all_usage_data_as_team_rows(period_start, period_end)
    teams = _get_teams_for_usage_reports()
    org_reports: Dict[str, OrgReport] = {}
    print('Generating reports for teams...')
    time_now = datetime.now()
    for team in teams:
        team_report = _get_team_report(all_data, team)
        _add_team_report_to_org_reports(org_reports, team, team_report, period_start)
    time_since = datetime.now() - time_now
    print(f'Generating reports for teams took {time_since.total_seconds()} seconds.')
    return org_reports

def _get_full_org_usage_report(org_report: OrgReport, instance_metadata: InstanceMetadata) -> FullUsageReport:
    if False:
        while True:
            i = 10
    return FullUsageReport(**dataclasses.asdict(org_report), **dataclasses.asdict(instance_metadata))

def _get_full_org_usage_report_as_dict(full_report: FullUsageReport) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return dataclasses.asdict(full_report)

@app.task(ignore_result=True, max_retries=3, autoretry_for=(Exception,))
def send_all_org_usage_reports(dry_run: bool=False, at: Optional[str]=None, capture_event_name: Optional[str]=None, skip_capture_event: bool=False, only_organization_id: Optional[str]=None) -> None:
    if False:
        return 10
    capture_event_name = capture_event_name or 'organization usage report'
    at_date = parser.parse(at) if at else None
    period = get_previous_day(at=at_date)
    (period_start, period_end) = period
    instance_metadata = get_instance_metadata(period)
    try:
        org_reports = _get_all_org_reports(period_start, period_end)
        print('Sending usage reports to PostHog and Billing...')
        time_now = datetime.now()
        for org_report in org_reports.values():
            org_id = org_report.organization_id
            if only_organization_id and only_organization_id != org_id:
                continue
            full_report = _get_full_org_usage_report(org_report, instance_metadata)
            full_report_dict = _get_full_org_usage_report_as_dict(full_report)
            if dry_run:
                continue
            if not skip_capture_event:
                at_date_str = at_date.isoformat() if at_date else None
                capture_report.delay(capture_event_name, org_id, full_report_dict, at_date_str)
            if has_non_zero_usage(full_report):
                send_report_to_billing_service.delay(org_id, full_report_dict)
        time_since = datetime.now() - time_now
        print(f'Sending usage reports to PostHog and Billing took {time_since.total_seconds()} seconds.')
    except Exception as err:
        capture_exception(err)
        raise err