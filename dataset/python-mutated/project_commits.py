from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint, ProjectReleasePermission
from sentry.api.paginator import OffsetPaginator
from sentry.api.serializers import serialize
from sentry.models.commit import Commit

@region_silo_endpoint
class ProjectCommitsEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}
    permission_classes = (ProjectReleasePermission,)

    def get(self, request: Request, project) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        List a Project\'s Commits\n        `````````````````````````\n\n        Retrieve a list of commits for a given project.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          commit belongs to.\n        :pparam string project_slug: the slug of the project to list the\n                                     commits of.\n        :qparam string query: this parameter can be used to create a\n                              "starts with" filter for the commit key.\n        '
        query = request.GET.get('query')
        queryset = Commit.objects.filter(organization_id=project.organization_id, releasecommit__release__releaseproject__project_id=project.id)
        if query:
            queryset = queryset.filter(key__istartswith=query)
        return self.paginate(request=request, queryset=queryset, order_by=('key', '-date_added') if query else '-date_added', on_results=lambda x: serialize(x, request.user), paginator_cls=OffsetPaginator)