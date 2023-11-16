from rest_framework.request import Request
from rest_framework.response import Response
from sentry import eventstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases import OrganizationEventsEndpointBase
from sentry.api.serializers import serialize
from sentry.api.serializers.models.event import SqlFormatEventSerializer
from sentry.constants import ObjectStatus
from sentry.models.project import Project

@region_silo_endpoint
class OrganizationEventDetailsEndpoint(OrganizationEventsEndpointBase):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, project_slug, event_id) -> Response:
        if False:
            for i in range(10):
                print('nop')
        'event_id is validated by a regex in the URL'
        if not self.has_feature(organization, request):
            return Response(status=404)
        try:
            project = Project.objects.get(slug=project_slug, organization_id=organization.id, status=ObjectStatus.ACTIVE)
        except Project.DoesNotExist:
            return Response(status=404)
        if not request.access.has_project_access(project):
            return Response(status=404)
        with self.handle_query_errors():
            event = eventstore.backend.get_event_by_id(project.id, event_id)
        if event is None:
            return Response({'detail': 'Event not found'}, status=404)
        if hasattr(event, 'for_group') and event.group:
            event = event.for_group(event.group)
        data = serialize(event, request.user, SqlFormatEventSerializer())
        data['projectSlug'] = project_slug
        return Response(data)