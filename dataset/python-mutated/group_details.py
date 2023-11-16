import functools
import logging
from datetime import timedelta
from typing import Sequence
from django.utils import timezone
from rest_framework.exceptions import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import features, tagstore, tsdb
from sentry.api import client
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import EnvironmentMixin, region_silo_endpoint
from sentry.api.bases import GroupEndpoint
from sentry.api.helpers.environments import get_environments
from sentry.api.helpers.group_index import delete_group_list, get_first_last_release, prep_search, update_groups
from sentry.api.serializers import GroupSerializer, GroupSerializerSnuba, serialize
from sentry.api.serializers.models.plugin import PluginSerializer, is_plugin_deprecated
from sentry.api.serializers.models.team import TeamSerializer
from sentry.issues.constants import get_issue_tsdb_group_model
from sentry.issues.escalating_group_forecast import EscalatingGroupForecast
from sentry.issues.grouptype import GroupCategory
from sentry.models.activity import Activity
from sentry.models.group import Group
from sentry.models.groupinbox import get_inbox_details
from sentry.models.groupowner import get_owner_details
from sentry.models.groupseen import GroupSeen
from sentry.models.groupsubscription import GroupSubscriptionManager
from sentry.models.team import Team
from sentry.models.userreport import UserReport
from sentry.plugins.base import plugins
from sentry.plugins.bases.issue2 import IssueTrackingPlugin2
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.tasks.post_process import fetch_buffered_group_stats
from sentry.types.ratelimit import RateLimit, RateLimitCategory
from sentry.utils import metrics
from sentry.utils.safe import safe_execute
delete_logger = logging.getLogger('sentry.deletions.api')

@region_silo_endpoint
class GroupDetailsEndpoint(GroupEndpoint, EnvironmentMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}
    enforce_rate_limit = True
    rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(5, 1), RateLimitCategory.USER: RateLimit(5, 1), RateLimitCategory.ORGANIZATION: RateLimit(5, 1)}, 'PUT': {RateLimitCategory.IP: RateLimit(5, 1), RateLimitCategory.USER: RateLimit(5, 1), RateLimitCategory.ORGANIZATION: RateLimit(5, 1)}, 'DELETE': {RateLimitCategory.IP: RateLimit(5, 5), RateLimitCategory.USER: RateLimit(5, 5), RateLimitCategory.ORGANIZATION: RateLimit(5, 5)}}

    def _get_activity(self, request: Request, group, num):
        if False:
            for i in range(10):
                print('nop')
        return Activity.objects.get_activities_for_group(group, num)

    def _get_seen_by(self, request: Request, group):
        if False:
            i = 10
            return i + 15
        seen_by = list(GroupSeen.objects.filter(group=group).order_by('-last_seen'))
        return serialize(seen_by, request.user)

    def _get_actions(self, request: Request, group):
        if False:
            print('Hello World!')
        project = group.project
        action_list = []
        for plugin in plugins.for_project(project, version=1):
            if is_plugin_deprecated(plugin, project):
                continue
            results = safe_execute(plugin.actions, request, group, action_list, _with_transaction=False)
            if not results:
                continue
            action_list = results
        for plugin in plugins.for_project(project, version=2):
            if is_plugin_deprecated(plugin, project):
                continue
            for action in safe_execute(plugin.get_actions, request, group, _with_transaction=False) or ():
                action_list.append(action)
        return action_list

    def _get_available_issue_plugins(self, request: Request, group):
        if False:
            return 10
        project = group.project
        plugin_issues = []
        for plugin in plugins.for_project(project, version=1):
            if isinstance(plugin, IssueTrackingPlugin2):
                if is_plugin_deprecated(plugin, project):
                    continue
                plugin_issues = safe_execute(plugin.plugin_issues, request, group, plugin_issues, _with_transaction=False)
        return plugin_issues

    def _get_context_plugins(self, request: Request, group):
        if False:
            print('Hello World!')
        project = group.project
        return serialize([plugin for plugin in plugins.for_project(project, version=None) if plugin.has_project_conf() and hasattr(plugin, 'get_custom_contexts') and plugin.get_custom_contexts()], request.user, PluginSerializer(project))

    @staticmethod
    def __group_hourly_daily_stats(group: Group, environment_ids: Sequence[int]):
        if False:
            for i in range(10):
                print('nop')
        get_range = functools.partial(tsdb.get_range, environment_ids=environment_ids, tenant_ids={'organization_id': group.project.organization_id})
        model = get_issue_tsdb_group_model(group.issue_category)
        now = timezone.now()
        hourly_stats = tsdb.rollup(get_range(model=model, keys=[group.id], end=now, start=now - timedelta(days=1)), 3600)[group.id]
        daily_stats = tsdb.rollup(get_range(model=model, keys=[group.id], end=now, start=now - timedelta(days=30)), 3600 * 24)[group.id]
        return (hourly_stats, daily_stats)

    @staticmethod
    def __get_group_global_count(group: Group) -> str:
        if False:
            return 10
        fetch_buffered_group_stats(group)
        return str(group.times_seen_with_pending)

    def get(self, request: Request, group) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve an Issue\n        `````````````````\n\n        Return details on an individual issue. This returns the basic stats for\n        the issue (title, last seen, first seen), some overall numbers (number\n        of comments, user reports) as well as the summarized event data.\n\n        :pparam string organization_slug: The slug of the organization.\n        :pparam string issue_id: the ID of the issue to retrieve.\n        :auth: required\n        '
        from sentry.utils import snuba
        try:
            organization = group.project.organization
            environments = get_environments(request, organization)
            environment_ids = [e.id for e in environments]
            expand = request.GET.getlist('expand', [])
            collapse = request.GET.getlist('collapse', [])
            data = serialize(group, request.user, GroupSerializerSnuba(environment_ids=environment_ids))
            activity = self._get_activity(request, group, num=100)
            seen_by = self._get_seen_by(request, group)
            if 'release' not in collapse:
                (first_release, last_release) = get_first_last_release(request, group)
                data.update({'firstRelease': first_release, 'lastRelease': last_release})
            if 'tags' not in collapse:
                tags = tagstore.get_group_tag_keys(group, environment_ids, limit=100, tenant_ids={'organization_id': group.project.organization_id})
                data.update({'tags': sorted(serialize(tags, request.user), key=lambda x: x['name'])})
            user_reports = UserReport.objects.filter(group_id=group.id) if not environment_ids else UserReport.objects.filter(group_id=group.id, environment_id__in=environment_ids)
            (hourly_stats, daily_stats) = self.__group_hourly_daily_stats(group, environment_ids)
            if 'inbox' in expand:
                inbox_map = get_inbox_details([group])
                inbox_reason = inbox_map.get(group.id)
                data.update({'inbox': inbox_reason})
            if 'owners' in expand:
                owner_details = get_owner_details([group], request.user)
                owners = owner_details.get(group.id)
                data.update({'owners': owners})
            if 'forecast' in expand and features.has('organizations:escalating-issues', group.organization):
                fetched_forecast = EscalatingGroupForecast.fetch(group.project_id, group.id)
                if fetched_forecast:
                    fetched_forecast = fetched_forecast.to_dict()
                    data.update({'forecast': {'data': fetched_forecast.get('forecast'), 'date_added': fetched_forecast.get('date_added')}})
            action_list = self._get_actions(request, group)
            data.update({'activity': serialize(activity, request.user), 'seenBy': seen_by, 'pluginActions': action_list, 'pluginIssues': self._get_available_issue_plugins(request, group), 'pluginContexts': self._get_context_plugins(request, group), 'userReportCount': user_reports.count(), 'stats': {'24h': hourly_stats, '30d': daily_stats}, 'count': self.__get_group_global_count(group)})
            participants = user_service.serialize_many(filter={'user_ids': GroupSubscriptionManager.get_participating_user_ids(group)}, as_user=request.user)
            for participant in participants:
                participant['type'] = 'user'
            if features.has('organizations:team-workflow-notifications', group.organization):
                team_ids = GroupSubscriptionManager.get_participating_team_ids(group)
                teams = Team.objects.filter(id__in=team_ids)
                team_serializer = TeamSerializer()
                serialized_teams = []
                for team in teams:
                    serialized_team = serialize(team, request.user, team_serializer)
                    serialized_team['type'] = 'team'
                    serialized_teams.append(serialized_team)
                participants.extend(serialized_teams)
            data.update({'participants': participants})
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 200, 'detail': 'group_details:get:response'})
            return Response(data)
        except snuba.RateLimitExceeded:
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 429, 'detail': 'group_details:get:snuba.RateLimitExceeded'})
            raise
        except Exception:
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 500, 'detail': 'group_details:get:Exception'})
            raise

    def put(self, request: Request, group) -> Response:
        if False:
            print('Hello World!')
        '\n        Update an Issue\n        ```````````````\n\n        Updates an individual issue\'s attributes. Only the attributes submitted\n        are modified.\n\n        :pparam string issue_id: the ID of the group to retrieve.\n        :param string status: the new status for the issue.  Valid values\n                              are ``"resolved"``, ``resolvedInNextRelease``,\n                              ``"unresolved"``, and ``"ignored"``.\n        :param string assignedTo: the user or team that should be assigned to\n                                  this issue. Can be of the form ``"<user_id>"``,\n                                  ``"user:<user_id>"``, ``"<username>"``,\n                                  ``"<user_primary_email>"``, or ``"team:<team_id>"``.\n        :param string assignedBy: ``"suggested_assignee"`` | ``"assignee_selector"``\n        :param boolean hasSeen: in case this API call is invoked with a user\n                                context this allows changing of the flag\n                                that indicates if the user has seen the\n                                event.\n        :param boolean isBookmarked: in case this API call is invoked with a\n                                     user context this allows changing of\n                                     the bookmark flag.\n        :param boolean isSubscribed:\n        :param boolean isPublic: sets the issue to public or private.\n        :param string substatus: the new substatus for the issues. Valid values\n                                 defined in GroupSubStatus.\n        :auth: required\n        '
        try:
            discard = request.data.get('discard')
            project = group.project
            search_fn = functools.partial(prep_search, self, request, project)
            response = update_groups(request, [group.id], [project], project.organization_id, search_fn)
            if discard or response.status_code != 200:
                return response
            group = Group.objects.get(id=group.id)
            serialized = serialize(group, request.user, GroupSerializer(environment_func=self._get_environment_func(request, group.project.organization_id)))
            return Response(serialized, status=response.status_code)
        except client.ApiError as e:
            logging.error('group_details:put client.ApiError', exc_info=True)
            return Response(e.body, status=e.status_code)
        except Exception:
            raise

    def delete(self, request: Request, group) -> Response:
        if False:
            while True:
                i = 10
        '\n        Remove an Issue\n        ```````````````\n\n        Removes an individual issue.\n\n        :pparam string issue_id: the ID of the issue to delete.\n        :auth: required\n        '
        from sentry.utils import snuba
        if group.issue_category != GroupCategory.ERROR:
            raise ValidationError(detail='Only error issues can be deleted.', code=400)
        try:
            delete_group_list(request, group.project, [group], 'delete')
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 200, 'detail': 'group_details:delete:Response'})
            return Response(status=202)
        except snuba.RateLimitExceeded:
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 429, 'detail': 'group_details:delete:snuba.RateLimitExceeded'})
            raise
        except Exception:
            metrics.incr('group.update.http_response', sample_rate=1.0, tags={'status': 500, 'detail': 'group_details:delete:Exception'})
            raise