from __future__ import annotations
from rest_framework.request import Request
from sentry.api.bases import OrganizationPermission
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.models.organization import Organization

class OrganizationRequestChangeEndpointPermission(OrganizationPermission):
    scope_map = {'POST': ['org:read']}

    def is_member_disabled_from_limit(self, request: Request, organization_or_id: Organization | int):
        if False:
            while True:
                i = 10
        return False

class OrganizationRequestChangeEndpoint(OrganizationEndpoint):
    permission_classes = (OrganizationRequestChangeEndpointPermission,)