from django.http import Http404
from django.utils.translation import gettext_lazy as _
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint, OrganizationIntegrationsLoosePermission
from sentry.api.paginator import OffsetPaginator
from sentry.api.serializers import serialize
from sentry.api.serializers.rest_framework.base import CamelSnakeModelSerializer
from sentry.models.integrations.repository_project_path_config import RepositoryProjectPathConfig
from sentry.models.project import Project
from sentry.models.repository import Repository
from sentry.services.hybrid_cloud.integration import integration_service

def gen_path_regex_field():
    if False:
        for i in range(10):
            print('nop')
    return serializers.RegexField('^[^\\s\'\\"]+$', required=True, allow_blank=True, error_messages={'invalid': _('Path may not contain spaces or quotations')})
BRANCH_NAME_ERROR_MESSAGE = 'Branch name may only have letters, numbers, underscores, forward slashes, dashes, and periods. Branch name may not start or end with a forward slash.'

class RepositoryProjectPathConfigSerializer(CamelSnakeModelSerializer):
    repository_id = serializers.IntegerField(required=True)
    project_id = serializers.IntegerField(required=True)
    stack_root = gen_path_regex_field()
    source_root = gen_path_regex_field()
    default_branch = serializers.RegexField('^(^(?![\\/]))([\\w\\.\\/-]+)(?<![\\/])$', required=True, error_messages={'invalid': _(BRANCH_NAME_ERROR_MESSAGE)})

    class Meta:
        model = RepositoryProjectPathConfig
        fields = ['repository_id', 'project_id', 'stack_root', 'source_root', 'default_branch']
        extra_kwargs = {}

    @property
    def org_integration(self):
        if False:
            while True:
                i = 10
        return self.context['organization_integration']

    @property
    def organization(self):
        if False:
            print('Hello World!')
        return self.context['organization']

    def validate(self, attrs):
        if False:
            return 10
        query = RepositoryProjectPathConfig.objects.filter(project_id=attrs.get('project_id'), stack_root=attrs.get('stack_root'))
        if self.instance:
            query = query.exclude(id=self.instance.id)
        if query.exists():
            raise serializers.ValidationError('Code path config already exists with this project and stack trace root')
        return attrs

    def validate_repository_id(self, repository_id):
        if False:
            while True:
                i = 10
        repo_query = Repository.objects.filter(id=repository_id, organization_id=self.organization.id)
        repo_query = repo_query.filter(integration_id=self.org_integration.integration_id)
        if not repo_query.exists():
            raise serializers.ValidationError('Repository does not exist')
        return repository_id

    def validate_project_id(self, project_id):
        if False:
            i = 10
            return i + 15
        project_query = Project.objects.filter(id=project_id, organization_id=self.organization.id)
        if not project_query.exists():
            raise serializers.ValidationError('Project does not exist')
        return project_id

    def create(self, validated_data):
        if False:
            while True:
                i = 10
        return RepositoryProjectPathConfig.objects.create(organization_integration_id=self.org_integration.id, organization_id=self.context['organization'].id, integration_id=self.context['organization_integration'].integration_id, **validated_data)

    def update(self, instance, validated_data):
        if False:
            while True:
                i = 10
        if 'id' in validated_data:
            validated_data.pop('id')
        for (key, value) in validated_data.items():
            setattr(self.instance, key, value)
        self.instance.save()
        return self.instance

class OrganizationIntegrationMixin:

    def get_organization_integration(self, organization, integration_id):
        if False:
            return 10
        org_integration = integration_service.get_organization_integration(integration_id=integration_id, organization_id=organization.id)
        if not org_integration:
            raise Http404
        return org_integration

    def get_project(self, organization, project_id):
        if False:
            return 10
        try:
            return Project.objects.get(organization=organization, id=project_id)
        except Project.DoesNotExist:
            raise Http404

@region_silo_endpoint
class OrganizationCodeMappingsEndpoint(OrganizationEndpoint, OrganizationIntegrationMixin):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'POST': ApiPublishStatus.UNKNOWN}
    permission_classes = (OrganizationIntegrationsLoosePermission,)

    def get(self, request: Request, organization) -> Response:
        if False:
            print('Hello World!')
        '\n        Get the list of repository project path configs\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team should be created for.\n        :qparam int integrationId: the optional integration id.\n        :qparam int project: Optional. Pass "-1" to filter to \'all projects user has access to\'. Omit to filter for \'all projects user is a member of\'.\n        :qparam int per_page: Pagination size.\n        :qparam string cursor: Pagination cursor.\n        :auth: required\n        '
        integration_id = request.GET.get('integrationId')
        queryset = RepositoryProjectPathConfig.objects.all()
        if integration_id:
            org_integration = self.get_organization_integration(organization, integration_id)
            queryset = queryset.filter(organization_integration_id=org_integration.id)
        else:
            projects = self.get_projects(request, organization)
            queryset = queryset.filter(project__in=projects)
        return self.paginate(request=request, queryset=queryset, on_results=lambda x: serialize(x, request.user), paginator_cls=OffsetPaginator)

    def post(self, request: Request, organization) -> Response:
        if False:
            print('Hello World!')
        '\n        Create a new repository project path config\n        ``````````````````\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team should be created for.\n        :param int repositoryId:\n        :param int projectId:\n        :param string stackRoot:\n        :param string sourceRoot:\n        :param string defaultBranch:\n        :param int required integrationId:\n        :auth: required\n        '
        integration_id = request.data.get('integrationId')
        if not integration_id:
            return self.respond('Missing param: integration_id', status=status.HTTP_400_BAD_REQUEST)
        try:
            project = Project.objects.get(id=request.data.get('projectId'))
        except Project.DoesNotExist:
            return self.respond('Could not find project', status=status.HTTP_404_NOT_FOUND)
        if not request.access.has_project_access(project):
            return self.respond(status=status.HTTP_403_FORBIDDEN)
        try:
            org_integration = self.get_organization_integration(organization, integration_id)
        except Http404:
            return self.respond('Could not find this integration installed on your organization', status=status.HTTP_404_NOT_FOUND)
        serializer = RepositoryProjectPathConfigSerializer(context={'organization': organization, 'organization_integration': org_integration}, data=request.data)
        if serializer.is_valid():
            repository_project_path_config = serializer.save()
            return self.respond(serialize(repository_project_path_config, request.user), status=status.HTTP_201_CREATED)
        return self.respond(serializer.errors, status=status.HTTP_400_BAD_REQUEST)