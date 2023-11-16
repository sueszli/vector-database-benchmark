from __future__ import annotations
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, List, TypedDict
from dateutil import parser
from django.db.models import F, Q
from django.http import HttpResponse
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import EnvironmentMixin, region_silo_endpoint
from sentry.api.bases.organization import OrganizationReleasesBaseEndpoint
from sentry.api.endpoints.release_thresholds.utils import get_errors_counts_timeseries_by_project_and_release
from sentry.api.serializers import serialize
from sentry.api.utils import get_date_range_from_params
from sentry.models.release import Release
from sentry.models.release_threshold.constants import ReleaseThresholdType, TriggerType
from sentry.services.hybrid_cloud.organization import RpcOrganization
from sentry.utils import metrics
logger = logging.getLogger('sentry.release_threshold_status')
if TYPE_CHECKING:
    from sentry.models.organization import Organization
    from sentry.models.project import Project
    from sentry.models.release_threshold.release_threshold import ReleaseThreshold

class SerializedThreshold(TypedDict):
    date: datetime
    environment: Dict[str, Any] | None
    project: Dict[str, Any]
    release: str
    threshold_type: int
    trigger_type: str
    value: int
    window_in_seconds: int

class EnrichedThreshold(SerializedThreshold):
    end: datetime
    is_healthy: bool
    key: str
    project_slug: str
    project_id: int
    start: datetime

class ReleaseThresholdStatusIndexSerializer(serializers.Serializer):
    start = serializers.DateTimeField(help_text='This defines the start of the time series range as an explicit datetime, either in UTC ISO8601 or epoch seconds.Use along with `end`', required=True)
    end = serializers.DateTimeField(help_text='This defines the inclusive end of the time series range as an explicit datetime, either in UTC ISO8601 or epoch seconds.Use along with `start`', required=True)
    environment = serializers.ListField(required=False, allow_empty=True, child=serializers.CharField(), help_text='Provide a list of environment names to filter your results by')
    project = serializers.ListField(required=False, allow_empty=True, child=serializers.CharField(), help_text='Provide a list of project slugs to filter your results by')
    release = serializers.ListField(required=False, allow_empty=True, child=serializers.CharField(), help_text='Provide a list of release versions to filter your results by')

    def validate(self, data):
        if False:
            i = 10
            return i + 15
        if data['start'] >= data['end']:
            raise serializers.ValidationError('Start datetime must be after End')
        return data

@region_silo_endpoint
class ReleaseThresholdStatusIndexEndpoint(OrganizationReleasesBaseEndpoint, EnvironmentMixin):
    owner: ApiOwner = ApiOwner.ENTERPRISE
    publish_status = {'GET': ApiPublishStatus.EXPERIMENTAL}

    def get(self, request: Request, organization: Organization | RpcOrganization) -> HttpResponse:
        if False:
            return 10
        "\n        List all derived statuses of releases that fall within the provided start/end datetimes\n\n        Constructs a response key'd off release_version, project_slug, environment, and lists thresholds with their status for *specified* projects\n        Each returned enriched threshold will contain the full serialized release_threshold instance as well as it's derived health status\n\n        {\n            {proj}-{env}-{release}: [\n                {\n                    project_id,\n                    project_slug,\n                    environment,\n                    ...,\n                    key: {release}-{proj}-{env},\n                    release_version: '',\n                    is_healthy: True/False,\n                    start: datetime,\n                    end: datetime,\n                },\n                {...},\n                {...}\n            ],\n            {proj}-{env}-{release}: [...],\n        }\n\n        ``````````````````\n\n        :param start: timestamp of the beginning of the specified date range\n        :param end: timestamp of the end of the specified date range\n\n        TODO:\n        - should we limit/paginate results? (this could get really bulky)\n        "
        data = request.data if len(request.GET) == 0 and hasattr(request, 'data') else request.GET
        start: datetime
        end: datetime
        (start, end) = get_date_range_from_params(params=data)
        logger.info('Checking release status health', extra={'start': start, 'end': end})
        metrics.incr('release.threshold_health_status.attempt')
        serializer = ReleaseThresholdStatusIndexSerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        environments_list = serializer.validated_data.get('environment')
        project_slug_list = serializer.validated_data.get('project')
        releases_list = serializer.validated_data.get('release')
        release_query = Q(organization=organization, date_added__gte=start, date_added__lte=end)
        if environments_list:
            release_query &= Q(releaseprojectenvironment__environment__name__in=environments_list)
        if project_slug_list:
            release_query &= Q(projects__slug__in=project_slug_list)
        if releases_list:
            release_query &= Q(version__in=releases_list)
        queryset = Release.objects.filter(release_query).annotate(date=F('date_added')).order_by('-date').distinct()
        queryset.prefetch_related('projects__release_thresholds')
        logger.info('Fetched releases', extra={'results': len(queryset), 'project_slugs': project_slug_list, 'releases': releases_list, 'environments': environments_list})
        thresholds_by_type: DefaultDict[int, dict[str, list]] = defaultdict()
        query_windows_by_type: DefaultDict[int, dict[str, datetime]] = defaultdict()
        for release in queryset:
            if project_slug_list:
                project_list = release.projects.filter(slug__in=project_slug_list)
            else:
                project_list = release.projects.all()
            for project in project_list:
                if environments_list:
                    thresholds_list: List[ReleaseThreshold] = project.release_thresholds.filter(environment__name__in=environments_list)
                else:
                    thresholds_list = project.release_thresholds.all()
                for threshold in thresholds_list:
                    if threshold.threshold_type not in thresholds_by_type:
                        thresholds_by_type[threshold.threshold_type] = {'project_ids': [], 'releases': [], 'thresholds': []}
                    thresholds_by_type[threshold.threshold_type]['project_ids'].append(project.id)
                    thresholds_by_type[threshold.threshold_type]['releases'].append(release.version)
                    if threshold.threshold_type not in query_windows_by_type:
                        query_windows_by_type[threshold.threshold_type] = {'start': datetime.now(tz=timezone.utc), 'end': datetime.now(tz=timezone.utc)}
                    query_windows_by_type[threshold.threshold_type]['start'] = min(release.date, query_windows_by_type[threshold.threshold_type]['start'])
                    query_windows_by_type[threshold.threshold_type]['end'] = max(release.date + timedelta(seconds=threshold.window_in_seconds), query_windows_by_type[threshold.threshold_type]['end'])
                    enriched_threshold: EnrichedThreshold = serialize(threshold)
                    enriched_threshold.update({'key': self.construct_threshold_key(release=release, project=project, threshold=threshold), 'start': release.date, 'end': release.date + timedelta(seconds=threshold.window_in_seconds), 'release': release.version, 'project_slug': project.slug, 'project_id': project.id, 'is_healthy': False})
                    thresholds_by_type[threshold.threshold_type]['thresholds'].append(enriched_threshold)
        release_threshold_health = defaultdict(list)
        for (threshold_type, filter_list) in thresholds_by_type.items():
            project_id_list = [proj_id for proj_id in filter_list['project_ids']]
            release_value_list = [release_version for release_version in filter_list['releases']]
            category_thresholds: List[EnrichedThreshold] = filter_list['thresholds']
            if threshold_type == ReleaseThresholdType.TOTAL_ERROR_COUNT:
                metrics.incr('release.threshold_health_status.check.error_count')
                "\n                Fetch errors timeseries for all projects with an error_count threshold in desired releases\n                Iterate through timeseries given threshold window and determine health status\n\n                NOTE: Timeseries query start & end are determined by API param window (_not_ threshold window)\n                    IF the param window doesn't cover the full threshold window, results will be inaccurate\n                TODO: If too many results, then throw an error and request user to narrow their search window\n                "
                query_window = query_windows_by_type[threshold_type]
                error_counts = get_errors_counts_timeseries_by_project_and_release(end=query_window['end'], environments_list=environments_list, organization_id=organization.id, project_id_list=project_id_list, release_value_list=release_value_list, start=query_window['start'])
                logger.info('querying error counts', extra={'start': query_window['start'], 'end': query_window['end'], 'project_ids': project_id_list, 'releases': release_value_list, 'environments': environments_list, 'error_count_data': error_counts})
                for ethreshold in category_thresholds:
                    is_healthy = is_error_count_healthy(ethreshold, error_counts)
                    ethreshold.update({'is_healthy': is_healthy})
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.NEW_ISSUE_COUNT:
                metrics.incr('release.threshold_health_status.check.new_issue_count')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.UNHANDLED_ISSUE_COUNT:
                metrics.incr('release.threshold_health_status.check.unhandled_issue_count')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.REGRESSED_ISSUE_COUNT:
                metrics.incr('release.threshold_health_status.check.regressed_issue_count')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.FAILURE_RATE:
                metrics.incr('release.threshold_health_status.check.failure_rate')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.CRASH_FREE_SESSION_RATE:
                metrics.incr('release.threshold_health_status.check.crash_free_session_rate')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
            elif threshold_type == ReleaseThresholdType.CRASH_FREE_USER_RATE:
                metrics.incr('release.threshold_health_status.check.crash_free_user_rate')
                for ethreshold in category_thresholds:
                    release_threshold_health[ethreshold['key']].append(ethreshold)
        return Response(release_threshold_health, status=200)

    def construct_threshold_key(self, project: Project, release: Release, threshold: ReleaseThreshold) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Consistent key helps to determine which thresholds can be grouped together.\n        project_slug - environment - release_version\n\n        NOTE: release versions can contain special characters... `-` delimiter may not be appropriate\n        NOTE: environment names can contain special characters... `-` delimiter may not be appropriate\n        TODO: move this into a separate helper?\n        '
        environment = threshold.environment.name if threshold.environment else 'None'
        return f'{project.slug}-{environment}-{release.version}'

def is_error_count_healthy(ethreshold: EnrichedThreshold, timeseries: List[Dict[str, Any]]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Iterate through timeseries given threshold window and determine health status\n    enriched threshold (ethreshold) includes `start`, `end`, and a constructed `key` identifier\n    '
    total_count = 0
    threshold_environment: str | None = ethreshold['environment']['name'] if ethreshold['environment'] else None
    sorted_series = sorted(timeseries, key=lambda x: x['time'])
    for i in sorted_series:
        if parser.parse(i['time']) > ethreshold['end']:
            logger.info('Reached end of threshold window. Breaking')
            metrics.incr('release.threshold_health_status.is_error_count_healthy.break_loop')
            break
        if parser.parse(i['time']) <= ethreshold['start'] or parser.parse(i['time']) > ethreshold['end'] or i['release'] != ethreshold['release'] or (i['project_id'] != ethreshold['project_id']) or (i['environment'] != threshold_environment):
            metrics.incr('release.threshold_health_status.is_error_count_healthy.skip')
            continue
        metrics.incr('release.threshold_health_status.is_error_count_healthy.aggregate_total')
        total_count += i['count()']
    logger.info('is_error_count_healthy', extra={'threshold': ethreshold, 'total_count': total_count, 'error_count_data': timeseries, 'threshold_environment': threshold_environment})
    if ethreshold['trigger_type'] == TriggerType.OVER_STR:
        return total_count <= ethreshold['value']
    return total_count >= ethreshold['value']