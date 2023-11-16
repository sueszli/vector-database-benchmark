from __future__ import annotations
import datetime
import uuid
from typing import Union
from rest_framework.exceptions import ParseError
from rest_framework.response import Response
from snuba_sdk import Column, Condition, Entity, Function, Granularity, Limit, Offset, Op, Or, OrderBy, Query, Request
from snuba_sdk.orderby import Direction
from sentry import features
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.event_search import ParenExpression, SearchFilter, parse_search_query
from sentry.api.paginator import GenericOffsetPaginator
from sentry.exceptions import InvalidSearchQuery
from sentry.models.project import Project
from sentry.replays.lib.new_query.errors import CouldNotParseValue, OperatorNotSupported
from sentry.replays.lib.new_query.fields import ColumnField
from sentry.replays.lib.query import attempt_compressed_condition
from sentry.replays.usecases.query import search_filter_to_condition
from sentry.replays.usecases.query.configs.scalar import click_search_config
from sentry.replays.usecases.query.fields import ComputedField, TagField
from sentry.utils.snuba import raw_snql_query
REFERRER = 'replays.query.query_replay_clicks_dataset'

@region_silo_endpoint
class ProjectReplayClicksIndexEndpoint(ProjectEndpoint):
    owner = ApiOwner.REPLAY
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project: Project, replay_id: str) -> Response:
        if False:
            i = 10
            return i + 15
        if not features.has('organizations:session-replay', project.organization, actor=request.user):
            return Response(status=404)
        filter_params = self.get_filter_params(request, project)
        try:
            replay_id = str(uuid.UUID(replay_id))
        except ValueError:
            return Response(status=404)

        def data_fn(offset, limit):
            if False:
                return 10
            try:
                search_filters = parse_search_query(request.query_params.get('query', ''))
            except InvalidSearchQuery as e:
                raise ParseError(str(e))
            return query_replay_clicks(project_id=filter_params['project_id'][0], replay_id=replay_id, start=filter_params['start'], end=filter_params['end'], limit=limit, offset=offset, search_filters=search_filters, organization_id=project.organization.id)
        return self.paginate(request=request, paginator=GenericOffsetPaginator(data_fn=data_fn), on_results=lambda results: {'data': results['data']})

def query_replay_clicks(project_id: int, replay_id: str, start: datetime.datetime, end: datetime.datetime, limit: int, offset: int, search_filters: SearchFilter, organization_id: int):
    if False:
        while True:
            i = 10
    'Query replay clicks.\n\n    This query is atypical in that it does not aggregate by replay_id and it is not exposed as a\n    user facing endpoint.  This query enables the replays client to fetch click information for\n    queries that were written for the replays index endpoint.  In other words, we need to translate\n    a list of conditions meant for an aggregated query into a list of conditions against a\n    non-aggregated query.  This means most of our ANDs become logical ORs and negation queries do\n    not logically filter any results.\n\n    Why do most ANDs become logical ORs?  Our query has been pre-validated to contain the result.\n    We know this replay matches the query now we just need to find the component parts that\n    created the match.  Because the filter (tag = "div" AND id = "button") works in an aggregated\n    context every row in the aggregation contributes to the result.  So in our query of a\n    pre-fetched result we know a single row could match both conditions or multiple rows could\n    match either condition independently.  Either case constitutes a successful response.  In the\n    case of selector matches those "AND" conditions will apply because they require a single row\n    matches all the conditions to produce the aggregated result set.\n\n    Why do negation queries have no impact?  Because if the aggregated result does not contain a\n    condition (e.g. tag = "button") then no row in the subset of the aggregation can logically\n    contain it.  We could remove these conditions but it is irrelevant to the output.  They are\n    logically disabled by the nature of the context they operate in.\n\n    If these conditions only apply to aggregated results why do we not aggregate here and simplify\n    our implementation?  Because aggregation precludes the ability to paginate.  There is no other\n    reason.\n    '
    conditions = handle_search_filters(click_search_config, search_filters)
    if len(conditions) > 1:
        conditions = [Or(conditions)]
    snuba_request = Request(dataset='replays', app_id='replay-backend-web', query=Query(match=Entity('replays'), select=[Function('identity', parameters=[Column('click_node_id')], alias='node_id'), Column('timestamp')], where=[Condition(Column('project_id'), Op.EQ, project_id), Condition(Column('timestamp'), Op.GTE, start), Condition(Column('timestamp'), Op.LT, end), Condition(Column('replay_id'), Op.EQ, replay_id), Condition(Column('click_tag'), Op.NEQ, ''), *conditions], orderby=[OrderBy(Column('timestamp'), Direction.ASC)], limit=Limit(limit), offset=Offset(offset), granularity=Granularity(3600)), tenant_ids={'organization_id': organization_id, 'referrer': 'replay-backend-web'})
    return raw_snql_query(snuba_request, REFERRER)

def handle_search_filters(search_config: dict[str, Union[ColumnField, ComputedField, TagField]], search_filters: list[Union[SearchFilter, str, ParenExpression]]) -> list[Condition]:
    if False:
        while True:
            i = 10
    'Convert search filters to snuba conditions.'
    result: list[Condition] = []
    look_back = None
    for search_filter in search_filters:
        if isinstance(search_filter, SearchFilter):
            try:
                condition = search_filter_to_condition(search_config, search_filter)
            except OperatorNotSupported:
                raise ParseError(f'Invalid operator specified for `{search_filter.key.name}`')
            except CouldNotParseValue:
                raise ParseError(f'Could not parse value for `{search_filter.key.name}`')
            if look_back == 'AND':
                look_back = None
                attempt_compressed_condition(result, condition, Or)
            elif look_back == 'OR':
                look_back = None
                attempt_compressed_condition(result, condition, Or)
            else:
                result.append(condition)
        elif isinstance(search_filter, ParenExpression):
            conditions = handle_search_filters(search_config, search_filter.children)
            if len(conditions) < 2:
                result.extend(conditions)
            else:
                result.append(Or(conditions))
        elif isinstance(search_filter, str):
            look_back = search_filter
    return result