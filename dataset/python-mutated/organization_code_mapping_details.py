from django.db.models.deletion import ProtectedError
from django.http import Http404
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint, OrganizationIntegrationsLoosePermission
from sentry.api.serializers import serialize
from sentry.models.integrations.repository_project_path_config import RepositoryProjectPathConfig
from sentry.services.hybrid_cloud.integration import integration_service
from .organization_code_mappings import OrganizationIntegrationMixin, RepositoryProjectPathConfigSerializer

@region_silo_endpoint
class OrganizationCodeMappingDetailsEndpoint(OrganizationEndpoint, OrganizationIntegrationMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}
    permission_classes = (OrganizationIntegrationsLoosePermission,)

    def convert_args(self, request: Request, organization_slug, config_id, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (args, kwargs) = super().convert_args(request, organization_slug, config_id, *args, **kwargs)
        ois = integration_service.get_organization_integrations(organization_id=kwargs['organization'].id)
        try:
            kwargs['config'] = RepositoryProjectPathConfig.objects.get(id=config_id, organization_integration_id__in=[oi.id for oi in ois])
        except RepositoryProjectPathConfig.DoesNotExist:
            raise Http404
        return (args, kwargs)

    def put(self, request: Request, config_id, organization, config) -> Response:
        if False:
            return 10
        '\n        Update a repository project path config\n        ``````````````````\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team should be created for.\n        :param int repository_id:\n        :param int project_id:\n        :param string stack_root:\n        :param string source_root:\n        :param string default_branch:\n        :auth: required\n        '
        try:
            org_integration = self.get_organization_integration(organization, config.integration_id)
        except Http404:
            return self.respond('Could not find this integration installed on your organization', status=status.HTTP_404_NOT_FOUND)
        serializer = RepositoryProjectPathConfigSerializer(context={'organization': organization, 'organization_integration': org_integration}, instance=config, data=request.data)
        if serializer.is_valid():
            repository_project_path_config = serializer.save()
            return self.respond(serialize(repository_project_path_config, request.user), status=status.HTTP_200_OK)
        return self.respond(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request: Request, config_id, organization, config) -> Response:
        if False:
            return 10
        '\n        Delete a repository project path config\n\n        :auth: required\n        '
        try:
            config.delete()
            return self.respond(status=status.HTTP_204_NO_CONTENT)
        except ProtectedError:
            return self.respond('Cannot delete Code Mapping. Must delete Code Owner that uses this mapping first.', status=status.HTTP_409_CONFLICT)