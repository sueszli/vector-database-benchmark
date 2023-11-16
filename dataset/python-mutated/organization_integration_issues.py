from typing import Any
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization_integrations import RegionOrganizationIntegrationBaseEndpoint
from sentry.integrations.mixins import IssueSyncMixin
from sentry.models.organization import Organization

@region_silo_endpoint
class OrganizationIntegrationIssuesEndpoint(RegionOrganizationIntegrationBaseEndpoint):
    publish_status = {'PUT': ApiPublishStatus.UNKNOWN}

    def put(self, request: Request, organization: Organization, integration_id: int, **kwds: Any) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Migrate plugin linked issues to integration linked issues\n        `````````````````````````````````````````````````````````\n        :pparam string organization: the organization the integration is installed in\n        :pparam string integration_id: the id of the integration\n        '
        integration = self.get_integration(organization.id, integration_id)
        install = integration.get_installation(organization_id=organization.id)
        if isinstance(install, IssueSyncMixin):
            install.migrate_issues()
            return Response(status=204)
        return Response(status=400)