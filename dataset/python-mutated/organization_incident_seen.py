from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.incident import IncidentEndpoint, IncidentPermission
from sentry.incidents.logic import set_incident_seen

@region_silo_endpoint
class OrganizationIncidentSeenEndpoint(IncidentEndpoint):
    publish_status = {'POST': ApiPublishStatus.UNKNOWN}
    permission_classes = (IncidentPermission,)

    def post(self, request: Request, organization, incident) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Mark an incident as seen by the user\n        ````````````````````````````````````\n\n        :auth: required\n        '
        set_incident_seen(incident=incident, user=request.user)
        return Response({}, status=201)