import functools
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import analytics, eventstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import EnvironmentMixin, region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint, ProjectEventPermission
from sentry.api.helpers.group_index import ValidationError, delete_groups, get_by_short_id, prep_search, track_slo_response, update_groups
from sentry.api.serializers import serialize
from sentry.api.serializers.models.group_stream import StreamGroupSerializer
from sentry.models.environment import Environment
from sentry.models.group import QUERY_STATUS_LOOKUP, Group, GroupStatus
from sentry.search.events.constants import EQUALITY_OPERATORS
from sentry.signals import advanced_search
from sentry.types.ratelimit import RateLimit, RateLimitCategory
from sentry.utils.validators import normalize_event_id
ERR_INVALID_STATS_PERIOD = "Invalid stats_period. Valid choices are '', '24h', and '14d'"

@region_silo_endpoint
class ProjectGroupIndexEndpoint(ProjectEndpoint, EnvironmentMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}
    permission_classes = (ProjectEventPermission,)
    enforce_rate_limit = True
    rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(5, 1), RateLimitCategory.USER: RateLimit(5, 1), RateLimitCategory.ORGANIZATION: RateLimit(5, 1)}}

    @track_slo_response('workflow')
    def get(self, request: Request, project) -> Response:
        if False:
            print('Hello World!')
        '\n        List a Project\'s Issues\n        ```````````````````````\n\n        Return a list of issues (groups) bound to a project.  All parameters are\n        supplied as query string parameters.\n\n        A default query of ``is:unresolved`` is applied. To return results\n        with other statuses send an new query value (i.e. ``?query=`` for all\n        results).\n\n        The ``statsPeriod`` parameter can be used to select the timeline\n        stats which should be present. Possible values are: ``""`` (disable),\n        ``"24h"``, ``"14d"``\n\n        :qparam string statsPeriod: an optional stat period (can be one of\n                                    ``"24h"``, ``"14d"``, and ``""``).\n        :qparam bool shortIdLookup: if this is set to true then short IDs are\n                                    looked up by this function as well.  This\n                                    can cause the return value of the function\n                                    to return an event issue of a different\n                                    project which is why this is an opt-in.\n                                    Set to `1` to enable.\n        :qparam querystring query: an optional Sentry structured search\n                                   query.  If not provided an implied\n                                   ``"is:unresolved"`` is assumed.)\n        :qparam string environment: this restricts the issues to ones containing\n                                    events from this environment\n        :pparam string organization_slug: the slug of the organization the\n                                          issues belong to.\n        :pparam string project_slug: the slug of the project the issues\n                                     belong to.\n        :auth: required\n        '
        stats_period = request.GET.get('statsPeriod')
        if stats_period not in (None, '', '24h', '14d'):
            return Response({'detail': ERR_INVALID_STATS_PERIOD}, status=400)
        elif stats_period is None:
            stats_period = '24h'
        elif stats_period == '':
            stats_period = None
        serializer = functools.partial(StreamGroupSerializer, environment_func=self._get_environment_func(request, project.organization_id), stats_period=stats_period)
        query = request.GET.get('query', '').strip()
        if query:
            matching_group = None
            matching_event = None
            event_id = normalize_event_id(query)
            if event_id:
                try:
                    matching_group = Group.objects.from_event_id(project, event_id)
                except Group.DoesNotExist:
                    pass
                else:
                    matching_event = eventstore.backend.get_event_by_id(project.id, event_id)
            elif matching_group is None:
                matching_group = get_by_short_id(project.organization_id, request.GET.get('shortIdLookup'), query)
                if matching_group is not None and matching_group.project_id != project.id:
                    matching_group = None
            if matching_group is not None:
                matching_event_environment = None
                try:
                    matching_event_environment = matching_event.get_environment().name if matching_event else None
                except Environment.DoesNotExist:
                    pass
                serialized_groups = serialize([matching_group], request.user, serializer())
                matching_event_id = getattr(matching_event, 'event_id', None)
                if matching_event_id:
                    serialized_groups[0]['matchingEventId'] = getattr(matching_event, 'event_id', None)
                if matching_event_environment:
                    serialized_groups[0]['matchingEventEnvironment'] = matching_event_environment
                response = Response(serialized_groups)
                response['X-Sentry-Direct-Hit'] = '1'
                return response
        try:
            (cursor_result, query_kwargs) = prep_search(self, request, project, {'count_hits': True})
        except ValidationError as exc:
            return Response({'detail': str(exc)}, status=400)
        results = list(cursor_result)
        context = serialize(results, request.user, serializer())
        status = [search_filter for search_filter in query_kwargs.get('search_filters', []) if search_filter.key.name == 'status' and search_filter.operator in EQUALITY_OPERATORS]
        if status and GroupStatus.UNRESOLVED in status[0].value.raw_value:
            status_labels = {QUERY_STATUS_LOOKUP[s] for s in status[0].value.raw_value}
            context = [r for r in context if 'status' not in r or r['status'] in status_labels]
        response = Response(context)
        self.add_cursor_headers(request, response, cursor_result)
        if results and query:
            advanced_search.send(project=project, sender=request.user)
            analytics.record('project_issue.searched', user_id=request.user.id, organization_id=project.organization_id, project_id=project.id, query=query)
        return response

    @track_slo_response('workflow')
    def put(self, request: Request, project) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Bulk Mutate a List of Issues\n        ````````````````````````````\n\n        Bulk mutate various attributes on issues.  The list of issues\n        to modify is given through the `id` query parameter.  It is repeated\n        for each issue that should be modified.\n\n        - For non-status updates, the `id` query parameter is required.\n        - For status updates, the `id` query parameter may be omitted\n          for a batch "update all" query.\n        - An optional `status` query parameter may be used to restrict\n          mutations to only events with the given status.\n\n        The following attributes can be modified and are supplied as\n        JSON object in the body:\n\n        If any IDs are out of scope this operation will succeed without\n        any data mutation.\n\n        :qparam int id: a list of IDs of the issues to be mutated.  This\n                        parameter shall be repeated for each issue.  It\n                        is optional only if a status is mutated in which\n                        case an implicit `update all` is assumed.\n        :qparam string status: optionally limits the query to issues of the\n                               specified status.  Valid values are\n                               ``"resolved"``, ``"unresolved"`` and\n                               ``"ignored"``.\n        :pparam string organization_slug: the slug of the organization the\n                                          issues belong to.\n        :pparam string project_slug: the slug of the project the issues\n                                     belong to.\n        :param string status: the new status for the issues.  Valid values\n                              are ``"resolved"``, ``"resolvedInNextRelease"``,\n                              ``"unresolved"``, and ``"ignored"``.\n        :param map statusDetails: additional details about the resolution.\n                                  Valid values are ``"inRelease"``, ``"inNextRelease"``,\n                                  ``"inCommit"``,  ``"ignoreDuration"``, ``"ignoreCount"``,\n                                  ``"ignoreWindow"``, ``"ignoreUserCount"``, and\n                                  ``"ignoreUserWindow"``.\n        :param int ignoreDuration: the number of minutes to ignore this issue.\n        :param boolean isPublic: sets the issue to public or private.\n        :param boolean merge: allows to merge or unmerge different issues.\n        :param string assignedTo: the user or team that should be assigned to\n                                  this issue. Can be of the form ``"<user_id>"``,\n                                  ``"user:<user_id>"``, ``"<username>"``,\n                                  ``"<user_primary_email>"``, or ``"team:<team_id>"``.\n        :param boolean hasSeen: in case this API call is invoked with a user\n                                context this allows changing of the flag\n                                that indicates if the user has seen the\n                                event.\n        :param boolean isBookmarked: in case this API call is invoked with a\n                                     user context this allows changing of\n                                     the bookmark flag.\n        :param string substatus: the new substatus for the issues. Valid values\n                                 defined in GroupSubStatus.\n        :auth: required\n        '
        search_fn = functools.partial(prep_search, self, request, project)
        return update_groups(request, request.GET.getlist('id'), [project], project.organization_id, search_fn)

    @track_slo_response('workflow')
    def delete(self, request: Request, project) -> Response:
        if False:
            print('Hello World!')
        "\n        Bulk Remove a List of Issues\n        ````````````````````````````\n\n        Permanently remove the given issues. The list of issues to\n        modify is given through the `id` query parameter.  It is repeated\n        for each issue that should be removed.\n\n        Only queries by 'id' are accepted.\n\n        If any IDs are out of scope this operation will succeed without\n        any data mutation.\n\n        :qparam int id: a list of IDs of the issues to be removed.  This\n                        parameter shall be repeated for each issue.\n        :pparam string organization_slug: the slug of the organization the\n                                          issues belong to.\n        :pparam string project_slug: the slug of the project the issues\n                                     belong to.\n        :auth: required\n        "
        search_fn = functools.partial(prep_search, self, request, project)
        return delete_groups(request, [project], project.organization_id, search_fn)