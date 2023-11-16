import json
from pathlib import Path
from typing import Dict
import structlog
from django.db.models import Q
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from rest_framework import request, response, serializers, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import SAFE_METHODS, BasePermission, IsAuthenticated
from rest_framework.request import Request
from posthog.api.forbid_destroy_model import ForbidDestroyModel
from posthog.api.routing import StructuredViewSetMixin
from posthog.models.dashboard_templates import DashboardTemplate
from posthog.permissions import ProjectMembershipNecessaryPermissions, TeamMemberAccessPermission
logger = structlog.get_logger(__name__)
dashboard_template_schema = json.loads((Path(__file__).parent / 'dashboard_template_schema.json').read_text())

class OnlyStaffCanEditDashboardTemplate(BasePermission):
    message = "You don't have edit permissions for this dashboard template."

    def has_permission(self, request: Request, view) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if request.method in SAFE_METHODS:
            return True
        return request.user.is_staff

class DashboardTemplateSerializer(serializers.ModelSerializer):

    class Meta:
        model = DashboardTemplate
        fields = ['id', 'template_name', 'dashboard_description', 'dashboard_filters', 'tags', 'tiles', 'variables', 'deleted', 'created_at', 'created_by', 'image_url', 'team_id', 'scope']

    def create(self, validated_data: Dict, *args, **kwargs) -> DashboardTemplate:
        if False:
            return 10
        if not validated_data['tiles']:
            raise ValidationError(detail='You need to provide tiles for the template.')
        if not validated_data.get('scope'):
            validated_data['scope'] = DashboardTemplate.Scope.ONLY_TEAM
        validated_data['team_id'] = self.context['team_id']
        return super().create(validated_data, *args, **kwargs)

    def update(self, instance: DashboardTemplate, validated_data: Dict, *args, **kwargs) -> DashboardTemplate:
        if False:
            i = 10
            return i + 15
        if validated_data.get('scope') == 'team' and instance.scope == 'global' and (not instance.team_id):
            raise ValidationError(detail='The original templates cannot be made private as they would be lost.')
        return super().update(instance, validated_data, *args, **kwargs)

class DashboardTemplateViewSet(StructuredViewSetMixin, ForbidDestroyModel, viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, ProjectMembershipNecessaryPermissions, TeamMemberAccessPermission, OnlyStaffCanEditDashboardTemplate]
    serializer_class = DashboardTemplateSerializer

    @method_decorator(cache_page(60 * 2))
    @action(methods=['GET'], detail=False)
    def json_schema(self, request: request.Request, **kwargs) -> response.Response:
        if False:
            i = 10
            return i + 15
        return response.Response(dashboard_template_schema)

    def get_queryset(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        filters = self.request.GET.dict()
        default_condition = Q(team_id=self.team_id) | Q(scope=DashboardTemplate.Scope.GLOBAL)
        return DashboardTemplate.objects.filter(Q(**filters) if filters else default_condition)