import sentry_sdk
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import features
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.reprocessing2 import CannotReprocess, pull_event_data

@region_silo_endpoint
class EventReprocessableEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project, event_id) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Retrieve information about whether an event can be reprocessed.\n        ```````````````````````````````````````````````````````````````\n\n        Returns `{"reprocessable": true}` if the event can be reprocessed, or\n        `{"reprocessable": false, "reason": "<code>"}` if it can\'t.\n\n        Returns 404 if the reprocessing feature is disabled.\n\n        Only entire issues can be reprocessed using\n        `GroupReprocessingEndpoint`, but we can tell you whether we will even\n        attempt to reprocess a particular event within that issue being\n        reprocessed based on what we know ahead of time.  reprocessable=true\n        means that the event may change in some way, reprocessable=false means\n        that there is no way it will change/improve.\n\n        Note this endpoint inherently suffers from time-of-check-time-of-use\n        problem (time of check=calling this endpoint, time of use=triggering\n        reprocessing) and the fact that event data + attachments is only\n        eventually consistent.\n\n        `<code>` can be:\n\n        * `unprocessed_event.not_found`: Can have many reasons. The event\n          is too old to be reprocessed (very unlikely!) or was not a native\n          event.\n        * `event.not_found`: The event does not exist.\n        * `attachment.not_found`: A required attachment, such as the original\n          minidump, is missing.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          issues belong to.\n        :pparam string project_slug: the slug of the project the event\n                                     belongs to.\n        :pparam string event_id: the id of the event.\n        :auth: required\n        '
        if not features.has('organizations:reprocessing-v2', project.organization, actor=request.user):
            return self.respond({'error': 'This project does not have the reprocessing v2 feature'}, status=404)
        try:
            pull_event_data(project.id, event_id)
        except CannotReprocess as e:
            sentry_sdk.set_tag('reprocessable', 'false')
            sentry_sdk.set_tag('reprocessable.reason', str(e))
            return self.respond({'reprocessable': False, 'reason': str(e)})
        else:
            sentry_sdk.set_tag('reprocessable', 'true')
            return self.respond({'reprocessable': True})