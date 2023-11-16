from datetime import timedelta
import sentry_sdk
from django.core.cache import cache
from django.utils import timezone
from rest_framework import serializers
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases import NoProjects, OrganizationEventsV2EndpointBase
from sentry.snuba import discover
from sentry.utils.hashlib import md5_text
MEASUREMENT_TYPES = {'web': ['measurements.fp', 'measurements.fcp', 'measurements.lcp', 'measurements.fid', 'measurements.cls'], 'mobile': ['measurements.app_start_cold', 'measurements.app_start_warm', 'measurements.frames_total', 'measurements.frames_slow', 'measurements.frames_frozen', 'measurements.stall_count', 'measurements.stall_stall_total_time', 'measurements.stall_stall_longest_time']}

class EventsHasMeasurementsQuerySerializer(serializers.Serializer):
    transaction = serializers.CharField(max_length=200)
    type = serializers.ChoiceField(choices=list(MEASUREMENT_TYPES.keys()))

    def validate(self, data):
        if False:
            print('Hello World!')
        project_ids = self.context.get('params', {}).get('project_id', [])
        if len(project_ids) != 1:
            raise serializers.ValidationError('Only 1 project allowed.')
        return data

@region_silo_endpoint
class OrganizationEventsHasMeasurementsEndpoint(OrganizationEventsV2EndpointBase):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization) -> Response:
        if False:
            while True:
                i = 10
        if not self.has_feature(organization, request):
            return Response(status=404)
        with sentry_sdk.start_span(op='discover.endpoint', description='parse params'):
            try:
                params = self.get_snuba_params(request, organization, check_global_views=False)
                now = timezone.now()
                params['start'] = now - timedelta(days=7)
                params['end'] = now
            except NoProjects:
                return Response({'measurements': False})
            data = {'transaction': request.GET.get('transaction'), 'type': request.GET.get('type')}
            serializer = EventsHasMeasurementsQuerySerializer(data=data, context={'params': params})
            if not serializer.is_valid():
                return Response(serializer.errors, status=400)
        org_id = organization.id
        project_id = params['project_id'][0]
        md5_hash = md5_text(data['transaction'], data['type']).hexdigest()
        cache_key = f'check-events-measurements:{org_id}:{project_id}:{md5_hash}'
        has_measurements = cache.get(cache_key)
        if has_measurements is None:
            with self.handle_query_errors():
                data = serializer.validated_data
                transaction_query = f"transaction:{data['transaction']}"
                measurements = MEASUREMENT_TYPES[data['type']]
                has_queries = [f'has:{measurement}' for measurement in measurements]
                measurement_query = ' OR '.join(has_queries)
                query = f'{transaction_query} ({measurement_query})'
                results = discover.query(selected_columns=['id'], query=query, params=params, limit=1, referrer='api.events.measurements', auto_fields=True, auto_aggregations=False, use_aggregate_conditions=False)
            has_measurements = len(results['data']) > 0
            cache.set(cache_key, has_measurements, 300)
        return Response({'measurements': has_measurements})