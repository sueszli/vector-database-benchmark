from typing import Any, Dict
from rest_framework.request import Request
from rest_framework.response import Response
from sentry_relay.processing import pii_selector_suggestions_from_event
from sentry import nodestore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.eventstore.models import Event

@region_silo_endpoint
class DataScrubbingSelectorSuggestionsEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization) -> Response:
        if False:
            print('Hello World!')
        '\n        Generate a list of data scrubbing selectors from existing event data.\n\n        This list is used to auto-complete settings in "Data Scrubbing" /\n        "Security and Privacy" settings.\n        '
        event_id = request.GET.get('eventId', None)
        projects = self.get_projects(request, organization)
        project_ids = [project.id for project in projects]
        suggestions: Dict[str, Any] = {}
        if event_id:
            node_ids = [Event.generate_node_id(p, event_id) for p in project_ids]
            all_data = nodestore.backend.get_multi(node_ids)
            data: Dict[str, Any]
            for data in filter(None, all_data.values()):
                for selector in pii_selector_suggestions_from_event(data):
                    examples_ = suggestions.setdefault(selector['path'], [])
                    if selector['value']:
                        examples_.append(selector['value'])
        return Response({'suggestions': [{'type': 'value', 'value': value, 'examples': examples} for (value, examples) in suggestions.items()]})