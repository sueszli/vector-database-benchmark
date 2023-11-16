from rest_framework.request import Request
from rest_framework.response import Response
from sentry import eventstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.models.project import Project
from sentry.types.ratelimit import RateLimit, RateLimitCategory
from sentry.utils.validators import INVALID_ID_DETAILS, is_event_id

@region_silo_endpoint
class EventIdLookupEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}
    enforce_rate_limit = True
    rate_limits = {'GET': {RateLimitCategory.IP: RateLimit(1, 1), RateLimitCategory.USER: RateLimit(1, 1), RateLimitCategory.ORGANIZATION: RateLimit(1, 1)}}

    def get(self, request: Request, organization, event_id) -> Response:
        if False:
            while True:
                i = 10
        '\n        Resolve an Event ID\n        ``````````````````\n\n        This resolves an event ID to the project slug and internal issue ID and internal event ID.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          event ID should be looked up in.\n        :param string event_id: the event ID to look up. validated by a\n                                regex in the URL.\n        :auth: required\n        '
        if event_id and (not is_event_id(event_id)):
            return Response({'detail': INVALID_ID_DETAILS.format('Event ID')}, status=400)
        project_slugs_by_id = dict(Project.objects.filter(organization=organization).values_list('id', 'slug'))
        try:
            snuba_filter = eventstore.Filter(conditions=[['event.type', '!=', 'transaction']], project_ids=list(project_slugs_by_id.keys()), event_ids=[event_id])
            event = eventstore.backend.get_events(filter=snuba_filter, limit=1, tenant_ids={'organization_id': organization.id})[0]
        except IndexError:
            raise ResourceDoesNotExist()
        else:
            return Response({'organizationSlug': organization.slug, 'projectSlug': project_slugs_by_id[event.project_id], 'groupId': str(event.group_id), 'eventId': str(event.event_id), 'event': serialize(event, request.user)})