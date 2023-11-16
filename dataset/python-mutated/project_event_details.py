from datetime import datetime
from typing import Any, List, Union
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import eventstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.serializers import IssueEventSerializer, serialize
from sentry.eventstore.models import Event, GroupEvent

def wrap_event_response(request_user: Any, event: Union[Event, GroupEvent], environments: List[str], include_full_release_data: bool=False):
    if False:
        i = 10
        return i + 15
    event_data = serialize(event, request_user, IssueEventSerializer(), include_full_release_data=include_full_release_data)
    next_event_id = None
    prev_event_id = None
    if event.group_id:
        conditions = []
        if environments:
            conditions.append(['environment', 'IN', environments])
        _filter = eventstore.Filter(conditions=conditions, project_ids=[event.project_id], group_ids=[event.group_id])
        (prev_ids, next_ids) = eventstore.backend.get_adjacent_event_ids(event, filter=_filter)
        next_event_id = next_ids[1] if next_ids else None
        prev_event_id = prev_ids[1] if prev_ids else None
    event_data['nextEventID'] = next_event_id
    event_data['previousEventID'] = prev_event_id
    return event_data

@region_silo_endpoint
class ProjectEventDetailsEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project, event_id) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve an Event for a Project\n        ```````````````````````````````\n\n        Return details on an individual event.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          event belongs to.\n        :pparam string project_slug: the slug of the project the event\n                                     belongs to.\n        :pparam string event_id: the id of the event to retrieve.\n                                 It is the hexadecimal id as\n                                 reported by the raven client)\n        :auth: required\n        '
        group_id = request.GET.get('group_id')
        group_id = int(group_id) if group_id else None
        event = eventstore.backend.get_event_by_id(project.id, event_id, group_id=group_id)
        if event is None:
            return Response({'detail': 'Event not found'}, status=404)
        environments = set(request.GET.getlist('environment'))
        if hasattr(event, 'for_group') and event.group:
            event = event.for_group(event.group)
        data = wrap_event_response(request.user, event, environments, include_full_release_data=True)
        return Response(data)
from rest_framework.request import Request
from rest_framework.response import Response

@region_silo_endpoint
class EventJsonEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project, event_id) -> Response:
        if False:
            return 10
        event = eventstore.backend.get_event_by_id(project.id, event_id)
        if not event:
            return Response({'detail': 'Event not found'}, status=404)
        event_dict = event.as_dict()
        if isinstance(event_dict['datetime'], datetime):
            event_dict['datetime'] = event_dict['datetime'].isoformat()
        return Response(event_dict, status=200)