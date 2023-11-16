from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.helpers.environments import environment_visibility_filter_options
from sentry.api.serializers import serialize
from sentry.models.environment import EnvironmentProject

@region_silo_endpoint
class ProjectEnvironmentsEndpoint(ProjectEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        List a Project\'s Environments\n        ```````````````````````````````\n\n        Return environments for a given project.\n\n        :qparam string visibility: when omitted only visible environments are\n                                   returned. Set to ``"hidden"`` for only hidden\n                                   environments, or ``"all"`` for both hidden\n                                   and visible environments.\n\n        :pparam string organization_slug: the slug of the organization the project\n                                          belongs to.\n\n        :pparam string project_slug: the slug of the project.\n\n        :auth: required\n        '
        queryset = EnvironmentProject.objects.filter(project=project, environment__organization_id=project.organization_id).exclude(environment__name='').select_related('environment').order_by('environment__name')
        visibility = request.GET.get('visibility', 'visible')
        if visibility not in environment_visibility_filter_options:
            return Response({'detail': "Invalid value for 'visibility', valid values are: {!r}".format(sorted(environment_visibility_filter_options.keys()))}, status=400)
        add_visibility_filters = environment_visibility_filter_options[visibility]
        queryset = add_visibility_filters(queryset)
        return Response(serialize(list(queryset), request.user))