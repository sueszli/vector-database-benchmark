from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.authentication import RelayAuthentication
from sentry.api.base import Endpoint, region_silo_endpoint
from sentry.api.permissions import RelayPermission
from sentry.models.projectkey import ProjectKey

@region_silo_endpoint
class RelayProjectIdsEndpoint(Endpoint):
    publish_status = {'POST': ApiPublishStatus.UNKNOWN}
    authentication_classes = (RelayAuthentication,)
    permission_classes = (RelayPermission,)
    owner = ApiOwner.OWNERS_INGEST

    def post(self, request: Request) -> Response:
        if False:
            while True:
                i = 10
        relay = request.relay
        assert relay is not None
        project_ids = {}
        for public_key in request.relay_request_data.get('publicKeys') or ():
            if not ProjectKey.looks_like_api_key(public_key):
                continue
            try:
                pk = ProjectKey.objects.get_from_cache(public_key=public_key)
            except ProjectKey.DoesNotExist:
                continue
            project_ids[public_key] = pk.project_id
        return Response({'projectIds': project_ids}, status=200)