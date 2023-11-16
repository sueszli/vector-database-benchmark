from uuid import uuid4
from django.db import router, transaction
from rest_framework import serializers, status
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import audit_log, features, roles
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.team import TeamEndpoint
from sentry.api.decorators import sudo_required
from sentry.api.fields.sentry_slug import SentrySlugField
from sentry.api.serializers import serialize
from sentry.api.serializers.models.team import TeamSerializer as ModelTeamSerializer
from sentry.api.serializers.rest_framework.base import CamelSnakeModelSerializer
from sentry.models.scheduledeletion import RegionScheduledDeletion
from sentry.models.team import Team, TeamStatus

class TeamSerializer(CamelSnakeModelSerializer):
    slug = SentrySlugField(max_length=50)
    org_role = serializers.ChoiceField(choices=tuple(list(roles.get_choices()) + ['']), default='')

    class Meta:
        model = Team
        fields = ('name', 'slug', 'org_role')

    def validate_slug(self, value):
        if False:
            for i in range(10):
                print('nop')
        qs = Team.objects.filter(slug=value, organization=self.instance.organization).exclude(id=self.instance.id)
        if qs.exists():
            raise serializers.ValidationError(f'The slug "{value}" is already in use.')
        return value

    def validate_org_role(self, value):
        if False:
            return 10
        if value == '':
            return None
        return value

@region_silo_endpoint
class TeamDetailsEndpoint(TeamEndpoint):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, team) -> Response:
        if False:
            while True:
                i = 10
        '\n        Retrieve a Team\n        ```````````````\n\n        Return details on an individual team.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team belongs to.\n        :pparam string team_slug: the slug of the team to get.\n        :qparam list expand: an optional list of strings to opt in to additional\n            data. Supports `projects`, `externalTeams`.\n        :qparam list collapse: an optional list of strings to opt out of certain\n            pieces of data. Supports `organization`.\n        :auth: required\n        '
        collapse = request.GET.getlist('collapse', [])
        expand = request.GET.getlist('expand', [])
        if 'organization' in collapse:
            collapse.remove('organization')
        else:
            expand.append('organization')
        return Response(serialize(team, request.user, ModelTeamSerializer(collapse=collapse, expand=expand)))

    def put(self, request: Request, team) -> Response:
        if False:
            return 10
        '\n        Update a Team\n        `````````````\n\n        Update various attributes and configurable settings for the given\n        team.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          team belongs to.\n        :pparam string team_slug: the slug of the team to get.\n        :param string name: the new name for the team.\n        :param string slug: a new slug for the team.  It has to be unique\n                            and available.\n        :param string orgRole: an organization role for the team. Only\n                               owners can set this value.\n        :auth: required\n        '
        team_org_role = team.org_role
        if team_org_role != request.data.get('orgRole'):
            if not features.has('organizations:org-roles-for-teams', team.organization, actor=None):
                del request.data['orgRole']
            if team.idp_provisioned:
                return Response({'detail': "This team is managed through your organization's identity provider."}, status=status.HTTP_403_FORBIDDEN)
            elif not request.access.has_scope('org:admin'):
                return Response({'detail': f'You must have the role of {roles.get_top_dog().id} to perform this action.'}, status=status.HTTP_403_FORBIDDEN)
        serializer = TeamSerializer(team, data=request.data, partial=True)
        if serializer.is_valid():
            team = serializer.save()
            data = team.get_audit_log_data()
            data['old_org_role'] = team_org_role
            self.create_audit_entry(request=request, organization=team.organization, target_object=team.id, event=audit_log.get_event_id('TEAM_EDIT'), data=data)
            return Response(serialize(team, request.user))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @sudo_required
    def delete(self, request: Request, team) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Delete a Team\n        `````````````\n\n        Schedules a team for deletion.\n\n        **Note:** Deletion happens asynchronously and therefore is not\n        immediate. Teams will have their slug released while waiting for deletion.\n        '
        suffix = uuid4().hex
        new_slug = f'{team.slug}-{suffix}'[0:50]
        try:
            with transaction.atomic(router.db_for_write(Team)):
                team = Team.objects.get(id=team.id, status=TeamStatus.ACTIVE)
                team.update(slug=new_slug, status=TeamStatus.PENDING_DELETION)
                scheduled = RegionScheduledDeletion.schedule(team, days=0, actor=request.user)
            self.create_audit_entry(request=request, organization=team.organization, target_object=team.id, event=audit_log.get_event_id('TEAM_REMOVE'), data=team.get_audit_log_data(), transaction_id=scheduled.id)
        except Team.DoesNotExist:
            pass
        return Response(status=204)