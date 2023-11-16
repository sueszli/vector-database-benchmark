from datetime import datetime
from typing import Dict, Optional
from zoneinfo import ZoneInfo
from rest_framework.exceptions import ValidationError
from ee.clickhouse.queries.experiments.trend_experiment_result import uses_math_aggregation_by_user_or_property_value
from posthog.constants import INSIGHT_FUNNELS, INSIGHT_TRENDS, TRENDS_CUMULATIVE
from posthog.models.feature_flag import FeatureFlag
from posthog.models.filters.filter import Filter
from posthog.models.team import Team
from posthog.queries.funnels import ClickhouseFunnel
from posthog.queries.trends.trends import Trends

class ClickhouseSecondaryExperimentResult:
    """
    This class calculates secondary metric values for Experiments.
    It returns value of metric for each variant.

    We adjust the metric filter based on Experiment parameters.
    """

    def __init__(self, filter: Filter, team: Team, feature_flag: FeatureFlag, experiment_start_date: datetime, experiment_end_date: Optional[datetime]=None):
        if False:
            while True:
                i = 10
        breakdown_key = f'$feature/{feature_flag.key}'
        self.variants = [variant['key'] for variant in feature_flag.variants]
        if team.timezone:
            start_date_in_project_timezone = experiment_start_date.astimezone(ZoneInfo(team.timezone))
            end_date_in_project_timezone = experiment_end_date.astimezone(ZoneInfo(team.timezone)) if experiment_end_date else None
        query_filter = filter.shallow_clone({'date_from': start_date_in_project_timezone, 'date_to': end_date_in_project_timezone, 'explicit_date': True, 'breakdown': breakdown_key, 'breakdown_type': 'event', 'properties': []})
        self.team = team
        if query_filter.insight == INSIGHT_TRENDS and (not uses_math_aggregation_by_user_or_property_value(query_filter)):
            query_filter = query_filter.shallow_clone({'display': TRENDS_CUMULATIVE})
        self.query_filter = query_filter

    def get_results(self):
        if False:
            return 10
        if self.query_filter.insight == INSIGHT_TRENDS:
            trend_results = Trends().run(self.query_filter, self.team)
            variants = self.get_trend_count_data_for_variants(trend_results)
        elif self.query_filter.insight == INSIGHT_FUNNELS:
            funnel_results = ClickhouseFunnel(self.query_filter, self.team).run()
            variants = self.get_funnel_conversion_rate_for_variants(funnel_results)
        else:
            raise ValidationError('Secondary metrics need to be funnel or trend insights')
        return {'result': variants}

    def get_funnel_conversion_rate_for_variants(self, insight_results) -> Dict[str, float]:
        if False:
            return 10
        variants = {}
        for result in insight_results:
            total = result[0]['count']
            success = result[-1]['count']
            breakdown_value = result[0]['breakdown_value'][0]
            if breakdown_value in self.variants:
                variants[breakdown_value] = round(int(success) / int(total), 3)
        return variants

    def get_trend_count_data_for_variants(self, insight_results) -> Dict[str, float]:
        if False:
            print('Hello World!')
        variants = {}
        for result in insight_results:
            count = result['count']
            breakdown_value = result['breakdown_value']
            if uses_math_aggregation_by_user_or_property_value(self.query_filter):
                count = result['count'] / len(result.get('data', [0]))
            if breakdown_value in self.variants:
                variants[breakdown_value] = count
        return variants