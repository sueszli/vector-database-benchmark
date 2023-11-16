from rest_framework.request import Request
from rest_framework.response import Response
from sentry import tsdb
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import EnvironmentMixin, StatsMixin, region_silo_endpoint
from sentry.api.bases.project import ProjectEndpoint
from sentry.api.exceptions import ResourceDoesNotExist
from sentry.ingest.inbound_filters import FILTER_STAT_KEYS_TO_VALUES
from sentry.models.environment import Environment
from sentry.tsdb.base import TSDBModel

@region_silo_endpoint
class ProjectStatsEndpoint(ProjectEndpoint, EnvironmentMixin, StatsMixin):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project) -> Response:
        if False:
            print('Hello World!')
        '\n        Retrieve Event Counts for a Project\n        ```````````````````````````````````\n\n        .. caution::\n           This endpoint may change in the future without notice.\n\n        Return a set of points representing a normalized timestamp and the\n        number of events seen in the period.\n\n        Query ranges are limited to Sentry\'s configured time-series\n        resolutions.\n\n        :pparam string organization_slug: the slug of the organization.\n        :pparam string project_slug: the slug of the project.\n        :qparam string stat: the name of the stat to query (``"received"``,\n                             ``"rejected"``, ``"blacklisted"``, ``generated``)\n        :qparam timestamp since: a timestamp to set the start of the query\n                                 in seconds since UNIX epoch.\n        :qparam timestamp until: a timestamp to set the end of the query\n                                 in seconds since UNIX epoch.\n        :qparam string resolution: an explicit resolution to search\n                                   for (one of ``10s``, ``1h``, and ``1d``)\n        :auth: required\n        '
        stat = request.GET.get('stat', 'received')
        query_kwargs = {}
        if stat == 'received':
            stat_model = TSDBModel.project_total_received
        elif stat == 'rejected':
            stat_model = TSDBModel.project_total_rejected
        elif stat == 'blacklisted':
            stat_model = TSDBModel.project_total_blacklisted
        elif stat == 'generated':
            stat_model = TSDBModel.project
            try:
                query_kwargs['environment_id'] = self._get_environment_id_from_request(request, project.organization_id)
            except Environment.DoesNotExist:
                raise ResourceDoesNotExist
        elif stat == 'forwarded':
            stat_model = TSDBModel.project_total_forwarded
        else:
            try:
                stat_model = FILTER_STAT_KEYS_TO_VALUES[stat]
            except KeyError:
                raise ValueError('Invalid stat: %s' % stat)
        data = tsdb.backend.get_range(model=stat_model, keys=[project.id], **self._parse_args(request, **query_kwargs), tenant_ids={'organization_id': project.organization_id})[project.id]
        return Response(data)