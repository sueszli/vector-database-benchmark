from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.models.group import Group

@region_silo_endpoint
class ShortIdLookupEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, short_id) -> Response:
        if False:
            while True:
                i = 10
        '\n        Resolve a Short ID\n        ``````````````````\n\n        This resolves a short ID to the project slug and internal issue ID.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          short ID should be looked up in.\n        :pparam string short_id: the short ID to look up.\n        :auth: required\n        '
        try:
            group = Group.objects.by_qualified_short_id(organization.id, short_id)
        except Group.DoesNotExist:
            raise ResourceDoesNotExist()
        return Response({'organizationSlug': organization.slug, 'projectSlug': group.project.slug, 'groupId': str(group.id), 'group': serialize(group, request.user), 'shortId': group.qualified_short_id})