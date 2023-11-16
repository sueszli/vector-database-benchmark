from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import audit_log
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.organization import ControlSiloOrganizationEndpoint, OrganizationAdminPermission
from sentry.api.serializers import serialize
from sentry.models.apikey import ApiKey
DEFAULT_SCOPES = ['project:read', 'event:read', 'team:read', 'org:read', 'member:read']

@control_silo_endpoint
class OrganizationApiKeyIndexEndpoint(ControlSiloOrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'POST': ApiPublishStatus.UNKNOWN}
    permission_classes = (OrganizationAdminPermission,)

    def get(self, request: Request, organization_context, organization) -> Response:
        if False:
            while True:
                i = 10
        "\n        List an Organization's API Keys\n        ```````````````````````````````````\n\n        :pparam string organization_slug: the organization short name\n        :auth: required\n        "
        queryset = sorted(ApiKey.objects.filter(organization_id=organization.id), key=lambda x: x.label)
        return Response(serialize(queryset, request.user))

    def post(self, request: Request, organization_context, organization) -> Response:
        if False:
            print('Hello World!')
        '\n        Create an Organization API Key\n        ```````````````````````````````````\n\n        :pparam string organization_slug: the organization short name\n        :auth: required\n        '
        key = ApiKey.objects.create(organization_id=organization.id, scope_list=DEFAULT_SCOPES)
        self.create_audit_entry(request, organization=organization, target_object=key.id, event=audit_log.get_event_id('APIKEY_ADD'), data=key.get_audit_log_data())
        return Response(serialize(key, request.user), status=status.HTTP_201_CREATED)