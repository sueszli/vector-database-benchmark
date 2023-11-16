import logging
import random
import string
from email.headerregistry import Address
from django.db import IntegrityError, router, transaction
from django.utils.text import slugify
from rest_framework.exceptions import NotAuthenticated, PermissionDenied
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.serializers import ValidationError
from sentry import audit_log, features
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint, OrganizationPermission
from sentry.api.endpoints.team_projects import ProjectPostSerializer
from sentry.api.exceptions import ConflictError, ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.experiments import manager as expt_manager
from sentry.models.organization import Organization
from sentry.models.organizationmember import OrganizationMember
from sentry.models.organizationmemberteam import OrganizationMemberTeam
from sentry.models.project import Project
from sentry.models.team import Team
from sentry.signals import project_created, team_created
from sentry.utils.snowflake import MaxSnowflakeRetryError
CONFLICTING_TEAM_SLUG_ERROR = 'A team with this slug already exists.'
MISSING_PERMISSION_ERROR_STRING = 'You do not have permission to join a new team as a Team Admin.'

def _generate_suffix() -> str:
    if False:
        print('Hello World!')
    letters = string.ascii_lowercase
    return ''.join((random.choice(letters) for _ in range(3)))

def fetch_slugifed_email_username(email: str) -> str:
    if False:
        print('Hello World!')
    return slugify(Address(addr_spec=email).username)

class OrgProjectPermission(OrganizationPermission):
    scope_map = {'POST': ['project:read', 'project:write', 'project:admin']}

@region_silo_endpoint
class OrganizationProjectsExperimentEndpoint(OrganizationEndpoint):
    publish_status = {'POST': ApiPublishStatus.EXPERIMENTAL}
    permission_classes = (OrgProjectPermission,)
    logger = logging.getLogger('team-project.create')
    owner = ApiOwner.ENTERPRISE

    def should_add_creator_to_team(self, request: Request):
        if False:
            while True:
                i = 10
        return request.user.is_authenticated

    def post(self, request: Request, organization: Organization) -> Response:
        if False:
            while True:
                i = 10
        "\n        Create a new Team and Project\n        ``````````````````\n\n        Create a new team where the user is set as Team Admin. The\n        name+slug of the team is automatically set as 'default-team-[username]'.\n        If this is taken, a random three letter suffix is added as needed\n        (eg: ...-gnm, ...-zls). Then create a new project bound to this team\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team should be created for.\n        :param string name: the name for the new project.\n        :param string platform: the optional platform that this project is for.\n        :param bool default_rules: create default rules (defaults to True)\n        :auth: required\n        "
        serializer = ProjectPostSerializer(data=request.data)
        if not serializer.is_valid():
            raise ValidationError(serializer.errors)
        if not self.should_add_creator_to_team(request):
            raise NotAuthenticated('User is not authenticated')
        result = serializer.validated_data
        exposed = expt_manager.get('ProjectCreationForAllExperimentV2', org=organization, actor=request.user)
        if not features.has('organizations:team-roles', organization) or not features.has('organizations:team-project-creation-all', organization) or exposed != 1:
            raise ResourceDoesNotExist(detail=MISSING_PERMISSION_ERROR_STRING)
        parsed_email = fetch_slugifed_email_username(request.user.email)
        project_name = result['name']
        default_team_slug = f'team-{parsed_email}'
        suffixed_team_slug = default_team_slug
        for _ in range(5):
            if not Team.objects.filter(organization=organization, slug=suffixed_team_slug).exists():
                break
            suffixed_team_slug = f'{default_team_slug}-{_generate_suffix()}'
        else:
            raise ConflictError({'detail': 'Unable to create a default team for this user. Please try again.'})
        default_team_slug = suffixed_team_slug
        try:
            with transaction.atomic(router.db_for_write(Team)):
                team = Team.objects.create(name=default_team_slug, slug=default_team_slug, idp_provisioned=result.get('idp_provisioned', False), organization=organization)
                member = OrganizationMember.objects.get(user_id=request.user.id, organization=organization)
                OrganizationMemberTeam.objects.create(team=team, organizationmember=member, role='admin')
                project = Project.objects.create(name=project_name, slug=None, organization=organization, platform=result.get('platform'))
        except (IntegrityError, MaxSnowflakeRetryError):
            raise ConflictError({'non_field_errors': [CONFLICTING_TEAM_SLUG_ERROR], 'detail': CONFLICTING_TEAM_SLUG_ERROR})
        except OrganizationMember.DoesNotExist:
            raise PermissionDenied(detail='You must be a member of the organization to join a new team as a Team Admin')
        else:
            project.add_team(team)
        team_created.send_robust(organization=organization, user=request.user, team=team, sender=self.__class__)
        self.create_audit_entry(request=request, organization=organization, target_object=team.id, event=audit_log.get_event_id('TEAM_ADD'), data=team.get_audit_log_data())
        self.create_audit_entry(request=request, organization=team.organization, target_object=project.id, event=audit_log.get_event_id('PROJECT_ADD'), data=project.get_audit_log_data())
        project_created.send(project=project, user=request.user, default_rules=result.get('default_rules', True), sender=self)
        self.create_audit_entry(request=request, organization=team.organization, event=audit_log.get_event_id('TEAM_AND_PROJECT_CREATED'), data={'team_slug': default_team_slug, 'project_slug': project_name})
        self.logger.info('created team through project creation flow', extra={'team_slug': default_team_slug, 'project_slug': project_name})
        serialized_response = serialize(project, request.user)
        serialized_response['team_slug'] = team.slug
        return Response(serialized_response, status=201)