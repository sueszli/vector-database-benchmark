from rest_framework.exceptions import ValidationError
from posthog.constants import FUNNEL_TO_STEP
from posthog.models.filters.filter import Filter
from posthog.models.team import Team
from posthog.queries.funnels.base import ClickhouseFunnelBase
from posthog.queries.funnels.utils import get_funnel_order_class

class ClickhouseFunnelTimeToConvert(ClickhouseFunnelBase):
    QUERY_TYPE = 'funnel_time_to_convert'

    def __init__(self, filter: Filter, team: Team) -> None:
        if False:
            while True:
                i = 10
        super().__init__(filter, team)
        self.funnel_order = get_funnel_order_class(filter)(filter, team)

    def _format_results(self, results: list) -> dict:
        if False:
            i = 10
            return i + 15
        return {'bins': [(bin_from_seconds, person_count) for (bin_from_seconds, person_count, _) in results], 'average_conversion_time': results[0][2]}

    def get_query(self) -> str:
        if False:
            print('Hello World!')
        steps_per_person_query = self.funnel_order.get_step_counts_query()
        self.params.update(self.funnel_order.params)
        from_step = self._filter.funnel_from_step or 0
        to_step = self._filter.funnel_to_step or len(self._filter.entities) - 1
        bin_count = self._filter.bin_count
        if bin_count is not None:
            if bin_count < 1:
                bin_count = 1
            elif bin_count > 90:
                bin_count = 90
            bin_count_identifier = str(bin_count)
            bin_count_expression = None
        else:
            bin_count_identifier = 'bin_count'
            bin_count_expression = f'\n                count() AS sample_count,\n                least(60, greatest(1, ceil(cbrt(ifNull(sample_count, 0))))) AS {bin_count_identifier},\n            '
        if not 0 < to_step < len(self._filter.entities):
            raise ValidationError(f"Filter parameter {FUNNEL_TO_STEP} can only be one of {', '.join(map(str, range(1, len(self._filter.entities))))} for time to convert!")
        steps_average_conversion_time_identifiers = [f'step_{step + 1}_average_conversion_time_inner' for step in range(from_step, to_step)]
        steps_average_conversion_time_expression_sum = ' + '.join(steps_average_conversion_time_identifiers)
        query = f"\n            WITH\n                step_runs AS (\n                    {steps_per_person_query}\n                ),\n                histogram_params AS (\n                    /* Binning ensures that each sample belongs to a bin in results */\n                    /* If bin_count is not a custom number, it's calculated in bin_count_expression */\n                    SELECT\n                        ifNull(floor(min({steps_average_conversion_time_expression_sum})), 0) AS from_seconds,\n                        ifNull(ceil(max({steps_average_conversion_time_expression_sum})), 1) AS to_seconds,\n                        round(avg({steps_average_conversion_time_expression_sum}), 2) AS average_conversion_time,\n                        {bin_count_expression or ''}\n                        ceil((to_seconds - from_seconds) / {bin_count_identifier}) AS bin_width_seconds_raw,\n                        /* Use 60 seconds as fallback bin width in case of only one sample */\n                        if(bin_width_seconds_raw > 0, bin_width_seconds_raw, 60) AS bin_width_seconds\n                    FROM step_runs\n                    -- We only need to check step to_step here, because it depends on all the other ones being NOT NULL too\n                    WHERE step_{to_step}_average_conversion_time_inner IS NOT NULL\n                ),\n                /* Below CTEs make histogram_params columns available to the query below as straightforward identifiers */\n                ( SELECT bin_width_seconds FROM histogram_params ) AS bin_width_seconds,\n                /* bin_count is only made available as an identifier if it had to be calculated */\n                {(f'( SELECT {bin_count_identifier} FROM histogram_params ) AS {bin_count_identifier},' if bin_count_expression else '')}\n                ( SELECT from_seconds FROM histogram_params ) AS histogram_from_seconds,\n                ( SELECT to_seconds FROM histogram_params ) AS histogram_to_seconds,\n                ( SELECT average_conversion_time FROM histogram_params ) AS histogram_average_conversion_time\n            SELECT\n                bin_from_seconds,\n                person_count,\n                histogram_average_conversion_time AS average_conversion_time\n            FROM (\n                /* Calculating bins from step runs */\n                SELECT\n                    histogram_from_seconds + floor(({steps_average_conversion_time_expression_sum} - histogram_from_seconds) / bin_width_seconds) * bin_width_seconds AS bin_from_seconds,\n                    count() AS person_count\n                FROM step_runs\n                GROUP BY bin_from_seconds\n            ) results\n            RIGHT OUTER JOIN (\n                /* Making sure bin_count bins are returned */\n                /* Those not present in the results query due to lack of data simply get person_count 0 */\n                SELECT histogram_from_seconds + number * bin_width_seconds AS bin_from_seconds FROM system.numbers LIMIT ifNull({bin_count_identifier}, 0) + 1\n            ) fill\n            USING (bin_from_seconds)\n            ORDER BY bin_from_seconds\n        "
        return query