from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationReleasesBaseEndpoint
from sentry.api.endpoints.project_release_file_details import ReleaseFileDetailsMixin
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.models.release import Release

class ReleaseFileSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200, required=True)

@region_silo_endpoint
class OrganizationReleaseFileDetailsEndpoint(OrganizationReleasesBaseEndpoint, ReleaseFileDetailsMixin):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, version, file_id) -> Response:
        if False:
            print('Hello World!')
        "\n        Retrieve an Organization Release's File\n        ```````````````````````````````````````\n\n        Return details on an individual file within a release.  This does\n        not actually return the contents of the file, just the associated\n        metadata.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :pparam string file_id: the ID of the file to retrieve.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        return self.get_releasefile(request, release, file_id, check_permission_fn=lambda : request.access.has_scope('project:write'))

    def put(self, request: Request, organization, version, file_id) -> Response:
        if False:
            for i in range(10):
                print('nop')
        "\n        Update an Organization Release's File\n        `````````````````````````````````````\n\n        Update metadata of an existing file.  Currently only the name of\n        the file can be changed.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :pparam string file_id: the ID of the file to update.\n        :param string name: the new name of the file.\n        :param string dist: the name of the dist.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        return self.update_releasefile(request, release, file_id)

    def delete(self, request: Request, organization, version, file_id) -> Response:
        if False:
            return 10
        "\n        Delete an Organization Release's File\n        `````````````````````````````````````\n\n        Permanently remove a file from a release.\n\n        This will also remove the physical file from storage.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :pparam string file_id: the ID of the file to delete.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        return self.delete_releasefile(release, file_id)