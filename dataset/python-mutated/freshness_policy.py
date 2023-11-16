import dagster._check as check
import graphene
from dagster._core.definitions.freshness_policy import FreshnessPolicy
from dagster._seven import get_current_datetime_in_utc, get_timestamp_from_utc_datetime

class GrapheneAssetFreshnessInfo(graphene.ObjectType):
    currentLagMinutes = graphene.Field(graphene.Float)
    currentMinutesLate = graphene.Field(graphene.Float)
    latestMaterializationMinutesLate = graphene.Field(graphene.Float)

    class Meta:
        name = 'AssetFreshnessInfo'

class GrapheneFreshnessPolicy(graphene.ObjectType):
    maximumLagMinutes = graphene.NonNull(graphene.Float)
    cronSchedule = graphene.Field(graphene.String)
    cronScheduleTimezone = graphene.Field(graphene.String)
    lastEvaluationTimestamp = graphene.Field(graphene.String)

    class Meta:
        name = 'FreshnessPolicy'

    def __init__(self, freshness_policy: FreshnessPolicy):
        if False:
            print('Hello World!')
        self._freshness_policy = check.inst_param(freshness_policy, 'freshness_policy', FreshnessPolicy)
        super().__init__(maximumLagMinutes=self._freshness_policy.maximum_lag_minutes, cronSchedule=self._freshness_policy.cron_schedule, cronScheduleTimezone=self._freshness_policy.cron_schedule_timezone)

    def resolve_lastEvaluationTimestamp(self, _graphene_info):
        if False:
            while True:
                i = 10
        tick = self._freshness_policy.get_evaluation_tick(get_current_datetime_in_utc())
        return str(int(get_timestamp_from_utc_datetime(tick) * 1000)) if tick else None