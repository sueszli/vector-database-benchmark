import logging
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationReleasesBaseEndpoint
from sentry.api.endpoints.project_release_files import ReleaseFilesMixin
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.models.release import Release

@region_silo_endpoint
class OrganizationReleaseFilesEndpoint(OrganizationReleasesBaseEndpoint, ReleaseFilesMixin):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'POST': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization, version) -> Response:
        if False:
            print('Hello World!')
        "\n        List an Organization Release's Files\n        ````````````````````````````````````\n\n        Retrieve a list of files for a given release.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :qparam string query: If set, only files with these partial names will be returned.\n        :qparam string checksum: If set, only files with these exact checksums will be returned.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        return self.get_releasefiles(request, release, organization.id)

    def post(self, request: Request, organization, version) -> Response:
        if False:
            for i in range(10):
                print('nop')
        "\n        Upload a New Organization Release File\n        ``````````````````````````````````````\n\n        Upload a new file for the given release.\n\n        Unlike other API requests, files must be uploaded using the\n        traditional multipart/form-data content-type.\n\n        The optional 'name' attribute should reflect the absolute path\n        that this file will be referenced as. For example, in the case of\n        JavaScript you might specify the full web URI.\n\n        :pparam string organization_slug: the slug of the organization the\n                                          release belongs to.\n        :pparam string version: the version identifier of the release.\n        :param string name: the name (full path) of the file.\n        :param file file: the multipart encoded file.\n        :param string dist: the name of the dist.\n        :param string header: this parameter can be supplied multiple times\n                              to attach headers to the file.  Each header\n                              is a string in the format ``key:value``.  For\n                              instance it can be used to define a content\n                              type.\n        :auth: required\n        "
        try:
            release = Release.objects.get(organization_id=organization.id, version=version)
        except Release.DoesNotExist:
            raise ResourceDoesNotExist
        logger = logging.getLogger('sentry.files')
        logger.info('organizationreleasefile.start')
        if not self.has_release_permission(request, organization, release):
            raise ResourceDoesNotExist
        return self.post_releasefile(request, release, logger)