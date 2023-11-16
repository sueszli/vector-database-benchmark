from __future__ import annotations
from rest_framework.request import Request
from sentry.api.bases import ProjectEndpoint, ProjectPermission
from sentry.services.hybrid_cloud.organization import RpcOrganization, RpcUserOrganizationContext

class ProjectRequestChangeEndpointPermission(ProjectPermission):
    scope_map = {'POST': ['org:read']}

    def is_member_disabled_from_limit(self, request: Request, organization: RpcOrganization | RpcUserOrganizationContext):
        if False:
            return 10
        return False

class ProjectRequestChangeEndpoint(ProjectEndpoint):
    permission_classes = (ProjectRequestChangeEndpointPermission,)