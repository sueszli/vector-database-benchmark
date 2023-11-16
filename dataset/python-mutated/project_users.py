from rest_framework.request import Request
from rest_framework.response import Response
from sentry import analytics
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.paginator import DateTimePaginator
from sentry.api.serializers import serialize
from sentry.models.eventuser import EventUser

@region_silo_endpoint
class ProjectUsersEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project) -> Response:
        if False:
            while True:
                i = 10
        "\n        List a Project's Users\n        ``````````````````````\n\n        Return a list of users seen within this project.\n\n        :pparam string organization_slug: the slug of the organization.\n        :pparam string project_slug: the slug of the project.\n        :pparam string key: the tag key to look up.\n        :auth: required\n        :qparam string query: Limit results to users matching the given query.\n                              Prefixes should be used to suggest the field to\n                              match on: ``id``, ``email``, ``username``, ``ip``.\n                              For example, ``query=email:foo@example.com``\n        "
        analytics.record('eventuser_endpoint.request', project_id=project.id, endpoint='sentry.api.endpoints.project_users.get')
        queryset = EventUser.objects.filter(project_id=project.id)
        if request.GET.get('query'):
            try:
                (field, identifier) = request.GET['query'].strip().split(':', 1)
                queryset = queryset.filter(project_id=project.id, **{EventUser.attr_from_keyword(field): identifier})
            except (ValueError, KeyError):
                return Response([])
        return self.paginate(request=request, queryset=queryset, order_by='-date_added', paginator_cls=DateTimePaginator, on_results=lambda x: serialize(x, request.user))