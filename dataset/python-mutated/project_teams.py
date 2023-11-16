from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.paginator import OffsetPaginator
from sentry.api.serializers import serialize
from sentry.models.team import Team

@region_silo_endpoint
class ProjectTeamsEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project) -> Response:
        if False:
            for i in range(10):
                print('nop')
        "\n        List a Project's Teams\n        ``````````````````````\n\n        Return a list of teams that have access to this project.\n\n        :pparam string organization_slug: the slug of the organization.\n        :pparam string project_slug: the slug of the project.\n        :auth: required\n        "
        queryset = Team.objects.filter(projectteam__project=project)
        return self.paginate(request=request, queryset=queryset, order_by='name', paginator_cls=OffsetPaginator, on_results=lambda x: serialize(x, request.user))