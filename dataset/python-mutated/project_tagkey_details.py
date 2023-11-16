from rest_framework.request import Request
from rest_framework.response import Response
from sentry import audit_log, tagstore
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import EnvironmentMixin, region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.constants import PROTECTED_TAG_KEYS
from sentry.models.environment import Environment
from sentry.types.ratelimit import RateLimit, RateLimitCategory

@region_silo_endpoint
class ProjectTagKeyDetailsEndpoint(ProjectEndpoint, EnvironmentMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN}
    enforce_rate_limit = True
    rate_limits = {'DELETE': {RateLimitCategory.IP: RateLimit(1, 1), RateLimitCategory.USER: RateLimit(1, 1), RateLimitCategory.ORGANIZATION: RateLimit(1, 1)}}

    def get(self, request: Request, project, key) -> Response:
        if False:
            return 10
        lookup_key = tagstore.prefix_reserved_key(key)
        try:
            environment_id = self._get_environment_id_from_request(request, project.organization_id)
        except Environment.DoesNotExist:
            raise ResourceDoesNotExist
        try:
            tagkey = tagstore.get_tag_key(project.id, environment_id, lookup_key, tenant_ids={'organization_id': project.organization_id})
        except tagstore.TagKeyNotFound:
            raise ResourceDoesNotExist
        return Response(serialize(tagkey, request.user))

    def delete(self, request: Request, project, key) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove all occurrences of the given tag key.\n\n            {method} {path}\n\n        '
        if key in PROTECTED_TAG_KEYS:
            return Response(status=403)
        lookup_key = tagstore.prefix_reserved_key(key)
        try:
            from sentry import eventstream
            eventstream_state = eventstream.backend.start_delete_tag(project.id, key)
            deleted = self.get_tag_keys_for_deletion(project, lookup_key)
            eventstream.backend.end_delete_tag(eventstream_state)
        except tagstore.TagKeyNotFound:
            raise ResourceDoesNotExist
        for tagkey in deleted:
            self.create_audit_entry(request=request, organization=project.organization, target_object=getattr(tagkey, 'id', None), event=audit_log.get_event_id('TAGKEY_REMOVE'), data=tagkey.get_audit_log_data())
        return Response(status=204)

    def get_tag_keys_for_deletion(self, project, key):
        if False:
            print('Hello World!')
        try:
            return [tagstore.get_tag_key(project_id=project.id, key=key, environment_id=None, tenant_ids={'organization_id': project.organization_id})]
        except tagstore.TagKeyNotFound:
            return []