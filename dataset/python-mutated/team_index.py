import logging
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.external_actor import ExternalActorEndpointMixin, ExternalTeamSerializer
from sentry.api.bases.team import TeamEndpoint
from sentry.api.serializers import serialize
from sentry.models.team import Team
logger = logging.getLogger(__name__)

@region_silo_endpoint
class ExternalTeamEndpoint(TeamEndpoint, ExternalActorEndpointMixin):
    publish_status = {'POST': ApiPublishStatus.UNKNOWN}

    def post(self, request: Request, team: Team) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Create an External Team\n        `````````````\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team belongs to.\n        :pparam string team_slug: the slug of the team to get.\n        :param required string provider: enum("github", "gitlab")\n        :param required string external_name: the associated Github/Gitlab team name.\n        :param optional string integration_id: the id of the integration if it exists.\n        :param string external_id: the associated user ID for this provider\n        :auth: required\n        '
        self.assert_has_feature(request, team.organization)
        serializer = ExternalTeamSerializer(data={**request.data, 'team_id': team.id}, context={'organization': team.organization})
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        (external_team, created) = serializer.save()
        status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
        return Response(serialize(external_team, request.user, key='team'), status=status_code)