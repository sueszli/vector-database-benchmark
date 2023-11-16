from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases import ProjectEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.models.grouphash import GroupHash
from sentry.models.grouptombstone import GroupTombstone

@region_silo_endpoint
class GroupTombstoneDetailsEndpoint(ProjectEndpoint):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN}

    def delete(self, request: Request, project, tombstone_id) -> Response:
        if False:
            print('Hello World!')
        '\n        Remove a GroupTombstone\n        ```````````````````````\n\n        Undiscards a group such that new events in that group will be captured.\n        This does not restore any previous data.\n\n        :pparam string organization_slug: the slug of the organization.\n        :pparam string project_slug: the slug of the project to which this tombstone belongs.\n        :pparam string tombstone_id: the ID of the tombstone to remove.\n        :auth: required\n        '
        try:
            tombstone = GroupTombstone.objects.get(project_id=project.id, id=tombstone_id)
        except GroupTombstone.DoesNotExist:
            raise ResourceDoesNotExist
        GroupHash.objects.filter(project_id=project.id, group_tombstone_id=tombstone_id).update(group_tombstone_id=None)
        tombstone.delete()
        return Response(status=204)