import copy
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, TypedDict, cast
import dateutil.parser
from django.db.models import Q
from django.utils import timezone
from sentry_sdk import capture_exception
from posthog.cache_utils import cache_for
from posthog.event_usage import report_organization_action
from posthog.models.organization import Organization, OrganizationUsageInfo
from posthog.models.team.team import Team
from posthog.redis import get_client
from posthog.tasks.usage_report import convert_team_usage_rows_to_dict, get_teams_with_billable_event_count_in_period, get_teams_with_recording_count_in_period
from posthog.utils import get_current_day
QUOTA_LIMITER_CACHE_KEY = '@posthog/quota-limits/'

class QuotaResource(Enum):
    EVENTS = 'events'
    RECORDINGS = 'recordings'
OVERAGE_BUFFER = {QuotaResource.EVENTS: 0, QuotaResource.RECORDINGS: 1000}

def replace_limited_team_tokens(resource: QuotaResource, tokens: Mapping[str, int]) -> None:
    if False:
        i = 10
        return i + 15
    pipe = get_client().pipeline()
    pipe.delete(f'{QUOTA_LIMITER_CACHE_KEY}{resource.value}')
    if tokens:
        pipe.zadd(f'{QUOTA_LIMITER_CACHE_KEY}{resource.value}', tokens)
    pipe.execute()

def add_limited_team_tokens(resource: QuotaResource, tokens: Mapping[str, int]) -> None:
    if False:
        for i in range(10):
            print('nop')
    redis_client = get_client()
    redis_client.zadd(f'{QUOTA_LIMITER_CACHE_KEY}{resource.value}', tokens)

def remove_limited_team_tokens(resource: QuotaResource, tokens: List[str]) -> None:
    if False:
        print('Hello World!')
    redis_client = get_client()
    redis_client.zrem(f'{QUOTA_LIMITER_CACHE_KEY}{resource.value}', *tokens)

@cache_for(timedelta(seconds=30), background_refresh=True)
def list_limited_team_tokens(resource: QuotaResource) -> List[str]:
    if False:
        return 10
    now = timezone.now()
    redis_client = get_client()
    results = redis_client.zrangebyscore(f'{QUOTA_LIMITER_CACHE_KEY}{resource.value}', min=now.timestamp(), max='+inf')
    return [x.decode('utf-8') for x in results]

class UsageCounters(TypedDict):
    events: int
    recordings: int

def org_quota_limited_until(organization: Organization, resource: QuotaResource) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    if not organization.usage:
        return None
    summary = organization.usage.get(resource.value, {})
    usage = summary.get('usage', 0)
    todays_usage = summary.get('todays_usage', 0)
    limit = summary.get('limit')
    if limit is None:
        return None
    is_quota_limited = usage + todays_usage >= limit + OVERAGE_BUFFER[resource]
    billing_period_end = round(dateutil.parser.isoparse(organization.usage['period'][1]).timestamp())
    if is_quota_limited and organization.never_drop_data:
        return None
    if is_quota_limited and billing_period_end:
        return billing_period_end
    return None

def sync_org_quota_limits(organization: Organization):
    if False:
        for i in range(10):
            print('nop')
    if not organization.usage:
        return None
    team_tokens: List[str] = [x for x in list(organization.teams.values_list('api_token', flat=True)) if x]
    if not team_tokens:
        capture_exception(Exception(f'quota_limiting: No team tokens found for organization: {organization.id}'))
        return
    for resource in [QuotaResource.EVENTS, QuotaResource.RECORDINGS]:
        quota_limited_until = org_quota_limited_until(organization, resource)
        if quota_limited_until:
            add_limited_team_tokens(resource, {x: quota_limited_until for x in team_tokens})
        else:
            remove_limited_team_tokens(resource, team_tokens)

def set_org_usage_summary(organization: Organization, new_usage: Optional[OrganizationUsageInfo]=None, todays_usage: Optional[UsageCounters]=None) -> bool:
    if False:
        print('Hello World!')
    has_changed = False
    new_usage = new_usage or cast(Optional[OrganizationUsageInfo], organization.usage)
    if not new_usage:
        return False
    new_usage = copy.deepcopy(new_usage)
    for field in ['events', 'recordings']:
        resource_usage = new_usage[field]
        if todays_usage:
            resource_usage['todays_usage'] = todays_usage[field]
        elif (organization.usage or {}).get(field, {}).get('usage') != resource_usage.get('usage'):
            resource_usage['todays_usage'] = 0
        else:
            resource_usage['todays_usage'] = organization.usage.get(field, {}).get('todays_usage') or 0
    has_changed = new_usage != organization.usage
    organization.usage = new_usage
    return has_changed

def update_all_org_billing_quotas(dry_run: bool=False) -> Dict[str, Dict[str, int]]:
    if False:
        print('Hello World!')
    period = get_current_day()
    (period_start, period_end) = period
    all_data = dict(teams_with_event_count_in_period=convert_team_usage_rows_to_dict(get_teams_with_billable_event_count_in_period(period_start, period_end)), teams_with_recording_count_in_period=convert_team_usage_rows_to_dict(get_teams_with_recording_count_in_period(period_start, period_end)))
    teams: Sequence[Team] = list(Team.objects.select_related('organization').exclude(Q(organization__for_internal_metrics=True) | Q(is_demo=True)))
    todays_usage_report: Dict[str, UsageCounters] = {}
    orgs_by_id: Dict[str, Organization] = {}
    for team in teams:
        team_report = UsageCounters(events=all_data['teams_with_event_count_in_period'].get(team.id, 0), recordings=all_data['teams_with_recording_count_in_period'].get(team.id, 0))
        org_id = str(team.organization.id)
        if org_id not in todays_usage_report:
            orgs_by_id[org_id] = team.organization
            todays_usage_report[org_id] = team_report.copy()
        else:
            org_report = todays_usage_report[org_id]
            for field in team_report:
                org_report[field] += team_report[field]
    quota_limited_orgs: Dict[str, Dict[str, int]] = {'events': {}, 'recordings': {}}
    for (org_id, todays_report) in todays_usage_report.items():
        org = orgs_by_id[org_id]
        if org.usage and org.usage.get('period'):
            if set_org_usage_summary(org, todays_usage=todays_report):
                org.save(update_fields=['usage'])
            for field in ['events', 'recordings']:
                quota_limited_until = org_quota_limited_until(org, QuotaResource(field))
                if quota_limited_until:
                    quota_limited_orgs[field][org_id] = quota_limited_until
    orgs_with_changes = set()
    previously_quota_limited_team_tokens: Dict[str, Dict[str, int]] = {'events': {}, 'recordings': {}}
    for field in quota_limited_orgs:
        previously_quota_limited_team_tokens[field] = list_limited_team_tokens(QuotaResource(field))
    quota_limited_teams: Dict[str, Dict[str, int]] = {'events': {}, 'recordings': {}}
    for team in teams:
        for field in quota_limited_orgs:
            org_id = str(team.organization.id)
            if org_id in quota_limited_orgs[field]:
                quota_limited_teams[field][team.api_token] = quota_limited_orgs[field][org_id]
                if team.api_token not in previously_quota_limited_team_tokens[field]:
                    orgs_with_changes.add(org_id)
            elif team.api_token in previously_quota_limited_team_tokens[field]:
                orgs_with_changes.add(org_id)
    for org_id in orgs_with_changes:
        properties = {'quota_limited_events': quota_limited_orgs['events'].get(org_id, None), 'quota_limited_recordings': quota_limited_orgs['events'].get(org_id, None)}
        report_organization_action(orgs_by_id[org_id], 'organization quota limits changed', properties=properties, group_properties=properties)
    if not dry_run:
        for field in quota_limited_teams:
            replace_limited_team_tokens(QuotaResource(field), quota_limited_teams[field])
    return quota_limited_orgs