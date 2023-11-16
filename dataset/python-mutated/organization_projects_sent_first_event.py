from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.api.serializers import serialize

@region_silo_endpoint
class OrganizationProjectsSentFirstEventEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Verify If Any Project Within An Organization Has Received a First Event\n        ```````````````````````````````````````````````````````````````````````\n\n        Returns true if any projects within the organization have received\n        a first event, false otherwise.\n\n        :pparam string organization_slug: the slug of the organization\n                                          containing the projects to check\n                                          for a first event from.\n        :qparam array[string] project:    An optional list of project ids to filter\n        :auth: required\n        '
        projects = self.get_projects(request, organization)
        seen_first_event = any((p.first_event for p in projects))
        return Response(serialize({'sentFirstEvent': seen_first_event}, request.user))