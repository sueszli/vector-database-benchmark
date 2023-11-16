from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationReleasesBaseEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.api.serializers import serialize
from sentry.models.release import Release
from sentry.models.releasecommit import ReleaseCommit

@region_silo_endpoint
class OrganizationReleaseCommitsEndpoint(OrganizationReleasesBaseEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, version) -> Response:
        if False:
            for i in range(10):
                print('nop')
        "\n        List an Organization Release's Commits\n        ``````````````````````````````````````\n\n        Retrieve a list of commits for a given release.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :auth: required\n        "
        try:
            release = Release.objects.distinct().get(organization_id=organization.id, projects__in=self.get_projects(request, organization), version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        queryset = ReleaseCommit.objects.filter(release=release).select_related('commit', 'commit__author')
        return self.paginate(request=request, queryset=queryset, order_by='order', on_results=lambda x: serialize([rc.commit for rc in x], request.user))