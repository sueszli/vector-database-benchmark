from datetime import datetime
from enum import Enum
from types import ModuleType
from typing import Dict, List, Optional, Sequence, TypedDict, Union, cast
import sentry_sdk
from rest_framework.exceptions import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases import OrganizationEventsV2EndpointBase
from sentry.models.organization import Organization
from sentry.search.events import fields
from sentry.snuba import discover, metrics_performance
from sentry.snuba.metrics.extraction import to_standard_metrics_query
from sentry.snuba.referrer import Referrer
from sentry.utils import metrics
from sentry.utils.snuba import SnubaTSResult

class CountResult(TypedDict):
    count: Optional[float]
MetricVolumeRow = List[Union[int, List[CountResult]]]

class StatsQualityEstimation(Enum):
    """
    Enum to represent the quality of the stats estimation
    """
    NO_DATA = 'no-data'
    NO_INDEXED_DATA = 'no-indexed-data'
    POOR_INDEXED_DATA = 'poor-indexed-data'
    ACCEPTABLE_INDEXED_DATA = 'acceptable-indexed-data'
    GOOD_INDEXED_DATA = 'good-indexed-data'

@region_silo_endpoint
class OrganizationMetricsEstimationStatsEndpoint(OrganizationEventsV2EndpointBase):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}
    "Gets the estimated volume of an organization's metric events."

    def get(self, request: Request, organization: Organization) -> Response:
        if False:
            i = 10
            return i + 15
        measurement = request.GET.get('yAxis')
        if measurement is None:
            return Response({'detail': 'missing required parameter yAxis'}, status=400)
        with sentry_sdk.start_span(op='discover.metrics.endpoint', description='get_full_metrics') as span:
            span.set_data('organization', organization)
            try:
                discover_stats = self.get_event_stats_data(request, organization, get_stats_generator(use_discover=True, remove_on_demand=False))
                stats_quality = estimate_stats_quality(discover_stats['data'])
                if _should_scale(measurement):
                    base_discover = self.get_event_stats_data(request, organization, get_stats_generator(use_discover=True, remove_on_demand=True))
                    base_metrics = self.get_event_stats_data(request, organization, get_stats_generator(use_discover=False, remove_on_demand=True))
                    estimated_volume = estimate_volume(discover_stats['data'], base_discover['data'], base_metrics['data'])
                    discover_stats['data'] = estimated_volume
                    if stats_quality == StatsQualityEstimation.NO_INDEXED_DATA and _count_non_zero_intervals(base_discover['data']) == 0:
                        stats_quality = StatsQualityEstimation.NO_DATA
                    metrics.incr('metrics_estimation_stats.data_quality', sample_rate=1.0, tags={'data_quality': stats_quality.value})
            except ValidationError:
                return Response({'detail': 'Comparison period is outside retention window'}, status=400)
        return Response(discover_stats, status=200)

def _count_non_zero_intervals(stats: List[MetricVolumeRow]) -> int:
    if False:
        return 10
    '\n    Counts the number of intervals with non-zero values\n    '
    non_zero_intervals = 0
    for idx in range(len(stats)):
        if _get_value(stats[idx]) != 0:
            non_zero_intervals += 1
    return non_zero_intervals

def estimate_stats_quality(stats: List[MetricVolumeRow]) -> StatsQualityEstimation:
    if False:
        i = 10
        return i + 15
    '\n    Estimates the quality of the stats estimation based on the number of intervals with no data\n    '
    if len(stats) == 0:
        return StatsQualityEstimation.NO_DATA
    data_intervals = _count_non_zero_intervals(stats)
    data_ratio = data_intervals / len(stats)
    if data_ratio >= 0.8:
        return StatsQualityEstimation.GOOD_INDEXED_DATA
    elif data_ratio > 0.4:
        return StatsQualityEstimation.ACCEPTABLE_INDEXED_DATA
    elif data_intervals > 0:
        return StatsQualityEstimation.POOR_INDEXED_DATA
    else:
        return StatsQualityEstimation.NO_INDEXED_DATA

def get_stats_generator(use_discover: bool, remove_on_demand: bool):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a get_stats function that can fetch from either metrics or discover and\n        with or without on_demand metrics.\n    '

    def get_discover_stats(query_columns: Sequence[str], query: str, params: Dict[str, str], rollup: int, zerofill_results: bool, comparison_delta: Optional[datetime]) -> SnubaTSResult:
        if False:
            for i in range(10):
                print('nop')
        if use_discover:
            module: ModuleType = discover
        else:
            module = metrics_performance
        if remove_on_demand:
            query = to_standard_metrics_query(query)
        return module.timeseries_query(selected_columns=query_columns, query=query, params=params, rollup=rollup, referrer=Referrer.API_ORGANIZATION_METRICS_ESTIMATION_STATS.value, zerofill_results=True, has_metrics=True)
    return get_discover_stats

def estimate_volume(indexed_data: List[MetricVolumeRow], base_index: List[MetricVolumeRow], base_metrics: List[MetricVolumeRow]) -> List[MetricVolumeRow]:
    if False:
        print('Hello World!')
    '\n    Estimates the volume of an on-demand metric by scaling the counts of the indexed metric with an estimated\n    sampling rate deduced from the factor of base_indexed and base_metrics time series.\n\n    The idea is that if we could multiply the indexed data by the actual sampling rate at each interval we would\n    obtain a good estimate of the volume. To get the actual sampling rate at any time we query both the indexed and\n    the metric data for the base metric (not the derived metric) and the ratio would be the approximate sample rate\n    '
    assert _is_data_aligned(indexed_data, base_index)
    assert _is_data_aligned(indexed_data, base_metrics)
    index_total = 0.0
    for elm in base_index:
        index_total += _get_value(elm)
    metrics_total = 0.0
    for elm in base_metrics:
        metrics_total += _get_value(elm)
    if index_total == 0.0:
        return indexed_data
    avg_inverted_rate = metrics_total / index_total
    for idx in range(len(indexed_data)):
        indexed = _get_value(base_index[idx])
        metrics = _get_value(base_metrics[idx])
        if indexed != 0:
            inverted_rate = metrics / indexed
        else:
            inverted_rate = avg_inverted_rate
        _set_value(indexed_data[idx], _get_value(indexed_data[idx]) * inverted_rate)
    return indexed_data

def _get_value(elm: MetricVolumeRow) -> float:
    if False:
        for i in range(10):
            print('nop')
    ret_val = cast(List[CountResult], elm[1])[0].get('count')
    if ret_val is None:
        return 0.0
    return ret_val

def _set_value(elm: MetricVolumeRow, value: float) -> None:
    if False:
        return 10
    cast(List[CountResult], elm[1])[0]['count'] = value

def _is_data_aligned(left: List[MetricVolumeRow], right: List[MetricVolumeRow]) -> bool:
    if False:
        return 10
    '\n    Checks if the two timeseries are aligned (represent the same time intervals).\n\n    Checks the length and the first and last timestamp (assumes they are correctly constructed, no\n    check for individual intervals)\n    '
    if len(left) != len(right):
        return False
    if len(left) == 0:
        return True
    return left[0][0] == right[0][0] and left[-1][0] == right[-1][0]

def _should_scale(metric: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Decides if the metric should be scaled ( based on the ratio between indexed and metrics data) or not\n\n    We can only scale counters ( percentiles and ratios cannot be scaled based on the ratio\n    between indexed and metrics data)\n    '
    if fields.is_function(metric):
        (function, params, alias) = fields.parse_function(metric)
        if function and function.lower() == 'count':
            return True
    return False