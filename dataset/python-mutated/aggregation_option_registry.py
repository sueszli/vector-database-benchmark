from enum import Enum
from typing import Optional
from sentry import options
from sentry.sentry_metrics.use_case_id_registry import UseCaseID, extract_use_case_id

class AggregationOption(Enum):
    HIST = 'hist'
    TEN_SECOND = 'ten_second'
METRIC_ID_AGG_OPTION = {'d:transactions/measurements.fcp@millisecond': AggregationOption.HIST, 'd:transactions/measurements.lcp@millisecond': AggregationOption.HIST}
USE_CASE_AGG_OPTION = {UseCaseID.CUSTOM: AggregationOption.TEN_SECOND}

def get_aggregation_option(metric_id: str) -> Optional[AggregationOption]:
    if False:
        while True:
            i = 10
    use_case_id: UseCaseID = extract_use_case_id(metric_id)
    if metric_id in METRIC_ID_AGG_OPTION:
        return METRIC_ID_AGG_OPTION.get(metric_id)
    elif options.get('sentry-metrics.10s-granularity') and use_case_id in USE_CASE_AGG_OPTION:
        return USE_CASE_AGG_OPTION[use_case_id]
    return None