import dataclasses
import urllib.parse
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TypedDict, Union, cast
from rest_framework.exceptions import ValidationError
from ee.clickhouse.queries.column_optimizer import EnterpriseColumnOptimizer
from ee.clickhouse.queries.groups_join_query import GroupsJoinQuery
from posthog.clickhouse.materialized_columns import get_materialized_columns
from posthog.constants import AUTOCAPTURE_EVENT, TREND_FILTER_TYPE_ACTIONS, FunnelCorrelationType
from posthog.models.element.element import chain_to_elements
from posthog.models.event.util import ElementSerializer
from posthog.models.filters import Filter
from posthog.models.property.util import get_property_string_expr
from posthog.models.team import Team
from posthog.models.team.team import groups_on_events_querying_enabled
from posthog.queries.funnels.utils import get_funnel_order_actor_class
from posthog.queries.insight import insight_sync_execute
from posthog.queries.person_distinct_id_query import get_team_distinct_ids_query
from posthog.queries.person_query import PersonQuery
from posthog.queries.util import correct_result_for_sampling
from posthog.utils import PersonOnEventsMode, generate_short_id

class EventDefinition(TypedDict):
    event: str
    properties: Dict[str, Any]
    elements: list

class EventOddsRatio(TypedDict):
    event: str
    success_count: int
    failure_count: int
    odds_ratio: float
    correlation_type: Literal['success', 'failure']

class EventOddsRatioSerialized(TypedDict):
    event: EventDefinition
    success_count: int
    success_people_url: Optional[str]
    failure_count: int
    failure_people_url: Optional[str]
    odds_ratio: float
    correlation_type: Literal['success', 'failure']

class FunnelCorrelationResponse(TypedDict):
    """
    The structure that the diagnose response will be returned in.
    NOTE: TypedDict is used here to comply with existing formats from other
    queries, but we could use, for example, a dataclass
    """
    events: List[EventOddsRatioSerialized]
    skewed: bool

@dataclasses.dataclass
class EventStats:
    success_count: int
    failure_count: int

@dataclasses.dataclass
class EventContingencyTable:
    """
    Represents a contingency table for a single event. Note that this isn't a
    complete contingency table, but rather only includes totals for
    failure/success as opposed to including the number of successes for cases
    that a persons _doesn't_ visit an event.
    """
    event: str
    visited: EventStats
    success_total: int
    failure_total: int

class FunnelCorrelation:
    TOTAL_IDENTIFIER = 'Total_Values_In_Query'
    ELEMENTS_DIVIDER = '__~~__'
    AUTOCAPTURE_EVENT_TYPE = '$event_type'
    MIN_PERSON_COUNT = 25
    MIN_PERSON_PERCENTAGE = 0.02
    PRIOR_COUNT = 1

    def __init__(self, filter: Filter, team: Team, base_uri: str='/') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._filter = filter
        self._team = team
        self._base_uri = base_uri
        if self._filter.funnel_step is None:
            self._filter = self._filter.shallow_clone({'funnel_step': 1})
        filter_data = {key: value for (key, value) in self._filter.to_dict().items() if not key.startswith('funnel_correlation_')}
        filter_data.update({'include_final_matching_events': self._filter.include_recordings})
        filter = Filter(data=filter_data, hogql_context=self._filter.hogql_context)
        funnel_order_actor_class = get_funnel_order_actor_class(filter)
        self._funnel_actors_generator = funnel_order_actor_class(filter, self._team, include_timestamp=True, include_preceding_timestamp=False, include_properties=self.properties_to_include)

    @property
    def properties_to_include(self) -> List[str]:
        if False:
            return 10
        props_to_include = []
        if self._team.person_on_events_mode != PersonOnEventsMode.DISABLED and self._filter.correlation_type == FunnelCorrelationType.PROPERTIES:
            mat_event_cols = get_materialized_columns('events')
            for property_name in cast(list, self._filter.correlation_property_names):
                if self._filter.aggregation_group_type_index is not None:
                    if not groups_on_events_querying_enabled():
                        continue
                    if '$all' == property_name:
                        return [f'group{self._filter.aggregation_group_type_index}_properties']
                    possible_mat_col = mat_event_cols.get((property_name, f'group{self._filter.aggregation_group_type_index}_properties'))
                    if possible_mat_col is not None:
                        props_to_include.append(possible_mat_col)
                    else:
                        props_to_include.append(f'group{self._filter.aggregation_group_type_index}_properties')
                else:
                    if '$all' == property_name:
                        return [f'person_properties']
                    possible_mat_col = mat_event_cols.get((property_name, 'person_properties'))
                    if possible_mat_col is not None:
                        props_to_include.append(possible_mat_col)
                    else:
                        props_to_include.append(f'person_properties')
        return props_to_include

    def support_autocapture_elements(self) -> bool:
        if False:
            while True:
                i = 10
        if self._filter.correlation_type == FunnelCorrelationType.EVENT_WITH_PROPERTIES and AUTOCAPTURE_EVENT in self._filter.correlation_event_names:
            return True
        return False

    def get_contingency_table_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Returns a query string and params, which are used to generate the contingency table.\n        The query returns success and failure count for event / property values, along with total success and failure counts.\n        '
        if self._filter.correlation_type == FunnelCorrelationType.PROPERTIES:
            return self.get_properties_query()
        if self._filter.correlation_type == FunnelCorrelationType.EVENT_WITH_PROPERTIES:
            return self.get_event_property_query()
        return self.get_event_query()

    def get_event_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            while True:
                i = 10
        (funnel_persons_query, funnel_persons_params) = self.get_funnel_actors_cte()
        event_join_query = self._get_events_join_query()
        query = f"\n            WITH\n                funnel_actors as ({funnel_persons_query}),\n                toDateTime(%(date_to)s, %(timezone)s) AS date_to,\n                toDateTime(%(date_from)s, %(timezone)s) AS date_from,\n                %(target_step)s AS target_step,\n                %(funnel_step_names)s as funnel_step_names\n\n            SELECT\n                event.event AS name,\n\n                -- If we have a `person.steps = target_step`, we know the person\n                -- reached the end of the funnel\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps = target_step\n                ) AS success_count,\n\n                -- And the converse being for failures\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps <> target_step\n                ) AS failure_count\n\n            FROM events AS event\n                {event_join_query}\n                AND event.event NOT IN %(exclude_event_names)s\n            GROUP BY name\n\n            -- To get the total success/failure numbers, we do an aggregation on\n            -- the funnel people CTE and count distinct actor_ids\n            UNION ALL\n\n            SELECT\n                -- We're not using WITH TOTALS because the resulting queries are\n                -- not runnable in Metabase\n                '{self.TOTAL_IDENTIFIER}' as name,\n\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps = target_step\n                ) AS success_count,\n\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps <> target_step\n                ) AS failure_count\n            FROM funnel_actors AS actors\n        "
        params = {**funnel_persons_params, 'funnel_step_names': self._get_funnel_step_names(), 'target_step': len(self._filter.entities), 'exclude_event_names': self._filter.correlation_event_exclude_names}
        return (query, params)

    def get_event_property_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            print('Hello World!')
        if not self._filter.correlation_event_names:
            raise ValidationError('Event Property Correlation expects atleast one event name to run correlation on')
        (funnel_persons_query, funnel_persons_params) = self.get_funnel_actors_cte()
        event_join_query = self._get_events_join_query()
        if self.support_autocapture_elements():
            (event_type_expression, _) = get_property_string_expr('events', self.AUTOCAPTURE_EVENT_TYPE, f"'{self.AUTOCAPTURE_EVENT_TYPE}'", 'properties')
            array_join_query = f"\n                'elements_chain' as prop_key,\n                concat({event_type_expression}, '{self.ELEMENTS_DIVIDER}', elements_chain) as prop_value,\n                tuple(prop_key, prop_value) as prop\n            "
        else:
            array_join_query = f"\n                arrayJoin(JSONExtractKeysAndValues(properties, 'String')) as prop\n            "
        query = f"\n            WITH\n                funnel_actors as ({funnel_persons_query}),\n                toDateTime(%(date_to)s, %(timezone)s) AS date_to,\n                toDateTime(%(date_from)s, %(timezone)s) AS date_from,\n                %(target_step)s AS target_step,\n                %(funnel_step_names)s as funnel_step_names\n\n            SELECT concat(event_name, '::', prop.1, '::', prop.2) as name,\n                   countDistinctIf(actor_id, steps = target_step) as success_count,\n                   countDistinctIf(actor_id, steps <> target_step) as failure_count\n            FROM (\n                SELECT\n                    actors.actor_id as actor_id,\n                    actors.steps as steps,\n                    events.event as event_name,\n                    -- Same as what we do in $all property queries\n                    {array_join_query}\n                FROM events AS event\n                    {event_join_query}\n                    AND event.event IN %(event_names)s\n            )\n            GROUP BY name\n            -- Discard high cardinality / low hits properties\n            -- This removes the long tail of random properties with empty, null, or very small values\n            HAVING (success_count + failure_count) > 2\n            AND prop.1 NOT IN %(exclude_property_names)s\n\n            UNION ALL\n            -- To get the total success/failure numbers, we do an aggregation on\n            -- the funnel people CTE and count distinct actor_ids\n            SELECT\n                '{self.TOTAL_IDENTIFIER}' as name,\n\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps = target_step\n                ) AS success_count,\n\n                countDistinctIf(\n                    actors.actor_id,\n                    actors.steps <> target_step\n                ) AS failure_count\n            FROM funnel_actors AS actors\n        "
        params = {**funnel_persons_params, 'funnel_step_names': self._get_funnel_step_names(), 'target_step': len(self._filter.entities), 'event_names': self._filter.correlation_event_names, 'exclude_property_names': self._filter.correlation_event_exclude_property_names}
        return (query, params)

    def get_properties_query(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        if not self._filter.correlation_property_names:
            raise ValidationError('Property Correlation expects atleast one Property to run correlation on')
        (funnel_actors_query, funnel_actors_params) = self.get_funnel_actors_cte()
        (person_prop_query, person_prop_params) = self._get_properties_prop_clause()
        (aggregation_join_query, aggregation_join_params) = self._get_aggregation_join_query()
        query = f"\n            WITH\n                funnel_actors as ({funnel_actors_query}),\n                %(target_step)s AS target_step\n            SELECT\n                concat(prop.1, '::', prop.2) as name,\n                -- We generate a unique identifier for each property value as: PropertyName::Value\n                countDistinctIf(actor_id, steps = target_step) AS success_count,\n                countDistinctIf(actor_id, steps <> target_step) AS failure_count\n            FROM (\n                SELECT\n                    actor_id,\n                    funnel_actors.steps as steps,\n                    /*\n                        We can extract multiple property values at the same time, since we're\n                        already querying the person table.\n                        This gives us something like:\n                        --------------------\n                        person1, steps, [property_value_0, property_value_1, property_value_2]\n                        person2, steps, [property_value_0, property_value_1, property_value_2]\n\n                        To group by property name, we need to extract the property from the array. ArrayJoin helps us do that.\n                        It transforms the above into:\n\n                        --------------------\n\n                        person1, steps, property_value_0\n                        person1, steps, property_value_1\n                        person1, steps, property_value_2\n\n                        person2, steps, property_value_0\n                        person2, steps, property_value_1\n                        person2, steps, property_value_2\n\n                        To avoid clashes and clarify the values, we also zip with the property name, to generate\n                        tuples like: (property_name, property_value), which we then group by\n                    */\n                    {person_prop_query}\n                FROM funnel_actors\n                {aggregation_join_query}\n\n            ) aggregation_target_with_props\n            -- Group by the tuple items: (property_name, property_value) generated by zip\n            GROUP BY prop.1, prop.2\n            HAVING prop.1 NOT IN %(exclude_property_names)s\n            UNION ALL\n            SELECT\n                '{self.TOTAL_IDENTIFIER}' as name,\n                countDistinctIf(actor_id, steps = target_step) AS success_count,\n                countDistinctIf(actor_id, steps <> target_step) AS failure_count\n            FROM funnel_actors\n        "
        params = {**funnel_actors_params, **person_prop_params, **aggregation_join_params, 'target_step': len(self._filter.entities), 'property_names': self._filter.correlation_property_names, 'exclude_property_names': self._filter.correlation_property_exclude_names}
        return (query, params)

    def _get_aggregation_target_join_query(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self._team.person_on_events_mode == PersonOnEventsMode.V1_ENABLED:
            aggregation_person_join = f'\n                JOIN funnel_actors as actors\n                    ON event.person_id = actors.actor_id\n            '
        else:
            aggregation_person_join = f"\n                JOIN ({get_team_distinct_ids_query(self._team.pk)}) AS pdi\n                        ON pdi.distinct_id = events.distinct_id\n\n                    -- NOTE: I would love to right join here, so we count get total\n                    -- success/failure numbers in one pass, but this causes out of memory\n                    -- error mentioning issues with right filling. I'm sure there's a way\n                    -- to do it but lifes too short.\n                    JOIN funnel_actors AS actors\n                        ON pdi.person_id = actors.actor_id\n                "
        aggregation_group_join = f'\n            JOIN funnel_actors AS actors\n                ON actors.actor_id = events.$group_{self._filter.aggregation_group_type_index}\n            '
        return aggregation_group_join if self._filter.aggregation_group_type_index is not None else aggregation_person_join

    def _get_events_join_query(self) -> str:
        if False:
            print('Hello World!')
        '\n        This query is used to join and filter the events table corresponding to the funnel_actors CTE.\n        It expects the following variables to be present in the CTE expression:\n            - funnel_actors\n            - date_to\n            - date_from\n            - funnel_step_names\n        '
        return f"\n            {self._get_aggregation_target_join_query()}\n\n            -- Make sure we're only looking at events before the final step, or\n            -- failing that, date_to\n            WHERE\n                -- add this condition in to ensure we can filter events before\n                -- joining funnel_actors\n                toTimeZone(toDateTime(event.timestamp), 'UTC') >= date_from\n                AND toTimeZone(toDateTime(event.timestamp), 'UTC') < date_to\n\n                AND event.team_id = {self._team.pk}\n\n                -- Add in per actor filtering on event time range. We just want\n                -- to include events that happened within the bounds of the\n                -- actors time in the funnel.\n                AND toTimeZone(toDateTime(event.timestamp), 'UTC') > actors.first_timestamp\n                AND toTimeZone(toDateTime(event.timestamp), 'UTC') < COALESCE(\n                    actors.final_timestamp,\n                    actors.first_timestamp + INTERVAL {self._funnel_actors_generator._filter.funnel_window_interval} {self._funnel_actors_generator._filter.funnel_window_interval_unit_ch()},\n                    date_to)\n                    -- Ensure that the event is not outside the bounds of the funnel conversion window\n\n                -- Exclude funnel steps\n                AND event.event NOT IN funnel_step_names\n        "

    def _get_aggregation_join_query(self):
        if False:
            print('Hello World!')
        if self._filter.aggregation_group_type_index is None:
            if self._team.person_on_events_mode != PersonOnEventsMode.DISABLED and groups_on_events_querying_enabled():
                return ('', {})
            (person_query, person_query_params) = PersonQuery(self._filter, self._team.pk, EnterpriseColumnOptimizer(self._filter, self._team.pk)).get_query()
            return (f'\n                JOIN ({person_query}) person\n                    ON person.id = funnel_actors.actor_id\n            ', person_query_params)
        else:
            return GroupsJoinQuery(self._filter, self._team.pk, join_key='funnel_actors.actor_id').get_join_query()

    def _get_properties_prop_clause(self):
        if False:
            return 10
        if self._team.person_on_events_mode != PersonOnEventsMode.DISABLED and groups_on_events_querying_enabled():
            group_properties_field = f'group{self._filter.aggregation_group_type_index}_properties'
            aggregation_properties_alias = 'person_properties' if self._filter.aggregation_group_type_index is None else group_properties_field
        else:
            group_properties_field = f'groups_{self._filter.aggregation_group_type_index}.group_properties_{self._filter.aggregation_group_type_index}'
            aggregation_properties_alias = PersonQuery.PERSON_PROPERTIES_ALIAS if self._filter.aggregation_group_type_index is None else group_properties_field
        if '$all' in cast(list, self._filter.correlation_property_names):
            return (f"\n                arrayJoin(JSONExtractKeysAndValues({aggregation_properties_alias}, 'String')) as prop\n            ", {})
        else:
            person_property_expressions = []
            person_property_params = {}
            for (index, property_name) in enumerate(cast(list, self._filter.correlation_property_names)):
                param_name = f'property_name_{index}'
                if self._filter.aggregation_group_type_index is not None:
                    (expression, _) = get_property_string_expr('groups' if self._team.person_on_events_mode == PersonOnEventsMode.DISABLED else 'events', property_name, f'%({param_name})s', aggregation_properties_alias, materialised_table_column=aggregation_properties_alias)
                else:
                    (expression, _) = get_property_string_expr('person' if self._team.person_on_events_mode == PersonOnEventsMode.DISABLED else 'events', property_name, f'%({param_name})s', aggregation_properties_alias, materialised_table_column=aggregation_properties_alias if self._team.person_on_events_mode != PersonOnEventsMode.DISABLED else 'properties')
                person_property_params[param_name] = property_name
                person_property_expressions.append(expression)
            return (f"\n                arrayJoin(arrayZip(\n                        %(property_names)s,\n                        [{','.join(person_property_expressions)}]\n                    )) as prop\n            ", person_property_params)

    def _get_funnel_step_names(self):
        if False:
            print('Hello World!')
        events: Set[Union[int, str]] = set()
        for entity in self._filter.entities:
            if entity.type == TREND_FILTER_TYPE_ACTIONS:
                action = entity.get_action()
                events.update(action.get_step_events())
            elif entity.id is not None:
                events.add(entity.id)
        return sorted(list(events))

    def _run(self) -> Tuple[List[EventOddsRatio], bool]:
        if False:
            print('Hello World!')
        '\n        Run the diagnose query.\n\n        Funnel Correlation queries take as input the same as the funnel query,\n        and returns the correlation of person events with a person successfully\n        getting to the end of the funnel. We use Odds Ratios as the correlation\n        metric. See https://en.wikipedia.org/wiki/Odds_ratio for more details.\n\n        Roughly speaking, to calculate the odds ratio, we build a contingency\n        table https://en.wikipedia.org/wiki/Contingency_table for each\n        dimension, then calculate the odds ratio for each.\n\n        For example, take for simplicity the cohort of all people, and the\n        success criteria of having a "signed up" event. First we would build a\n        contingency table like:\n\n        |                    | success | failure | total |\n        | -----------------: | :-----: | :-----: | :---: |\n        | watched video      |    5    |    1    |   6   |\n        | didn\'t watch video |    2    |   10    |   12  |\n\n\n        Then the odds that a person signs up given they watched the video is 5 /\n        1.\n\n        And the odds that a person signs up given they didn\'t watch the video is\n        2 / 10.\n\n        So we say the odds ratio is 5 / 1 over 2 / 10 = 25 . The further away the\n        odds ratio is from 1, the greater the correlation.\n\n        Requirements:\n\n         - Intitially we only need to consider the names of events that a cohort\n           person has emitted. So we explicitly are not interested in e.g.\n           correlating properties, although this will be a follow-up.\n\n        Non-functional requirements:\n\n         - there can be perhaps millions of people in a cohort, so we should\n           consider this when writing the algorithm. e.g. we should probably\n           avoid pulling all people into across the wire.\n         - there can be an order of magnitude more events than people, so we\n           should avoid pulling all events across the wire.\n         - there may be a large but not huge number of distinct events, let\'s say\n           100 different names for events. We should avoid n+1 queries for the\n           event names dimension\n\n        Contincency tables are something we can pull out of the db, so we can\n        have a query that:\n\n         1. filters people by the cohort criteria\n         2. groups these people by the success criteria\n         3. groups people by our criterion with which we want to test\n            correlation, e.g. "watched video"\n\n        '
        self._filter.team = self._team
        (event_contingency_tables, success_total, failure_total) = self.get_partial_event_contingency_tables()
        success_total = int(correct_result_for_sampling(success_total, self._filter.sampling_factor))
        failure_total = int(correct_result_for_sampling(failure_total, self._filter.sampling_factor))
        if not success_total or not failure_total:
            return ([], True)
        skewed_totals = False
        if success_total / failure_total > 10 or failure_total / success_total > 10:
            skewed_totals = True
        odds_ratios = [get_entity_odds_ratio(event_stats, FunnelCorrelation.PRIOR_COUNT) for event_stats in event_contingency_tables if not FunnelCorrelation.are_results_insignificant(event_stats)]
        positively_correlated_events = sorted([odds_ratio for odds_ratio in odds_ratios if odds_ratio['correlation_type'] == 'success'], key=lambda x: x['odds_ratio'], reverse=True)
        negatively_correlated_events = sorted([odds_ratio for odds_ratio in odds_ratios if odds_ratio['correlation_type'] == 'failure'], key=lambda x: x['odds_ratio'], reverse=False)
        events = positively_correlated_events[:10] + negatively_correlated_events[:10]
        return (events, skewed_totals)

    def construct_people_url(self, success: bool, event_definition: EventDefinition, cache_invalidation_key: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given an event_definition and success/failure flag, returns a url that\n        get be used to GET the associated people for the event/sucess pair. The\n        primary purpose of this is to reduce the risk of clients of the API\n        fetching incorrect people, given an event definition.\n        '
        if not self._filter.correlation_type or self._filter.correlation_type == FunnelCorrelationType.EVENTS:
            return self.construct_event_correlation_people_url(success=success, event_definition=event_definition, cache_invalidation_key=cache_invalidation_key)
        elif self._filter.correlation_type == FunnelCorrelationType.EVENT_WITH_PROPERTIES:
            return self.construct_event_with_properties_people_url(success=success, event_definition=event_definition, cache_invalidation_key=cache_invalidation_key)
        elif self._filter.correlation_type == FunnelCorrelationType.PROPERTIES:
            return self.construct_person_properties_people_url(success=success, event_definition=event_definition, cache_invalidation_key=cache_invalidation_key)
        return None

    def construct_event_correlation_people_url(self, success: bool, event_definition: EventDefinition, cache_invalidation_key: str) -> str:
        if False:
            print('Hello World!')
        params = self._filter.shallow_clone({'funnel_correlation_person_converted': 'true' if success else 'false', 'funnel_correlation_person_entity': {'id': event_definition['event'], 'type': 'events'}}).to_params()
        return f'{self._base_uri}api/person/funnel/correlation/?{urllib.parse.urlencode(params)}&cache_invalidation_key={cache_invalidation_key}'

    def construct_event_with_properties_people_url(self, success: bool, event_definition: EventDefinition, cache_invalidation_key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.support_autocapture_elements():
            (event_name, _, _) = event_definition['event'].split('::')
            elements = event_definition['elements']
            first_element = elements[0]
            elements_as_action = {'tag_name': first_element['tag_name'], 'href': first_element['href'], 'text': first_element['text'], 'selector': build_selector(elements)}
            params = self._filter.shallow_clone({'funnel_correlation_person_converted': 'true' if success else 'false', 'funnel_correlation_person_entity': {'id': event_name, 'type': 'events', 'properties': [{'key': property_key, 'value': [property_value], 'type': 'element', 'operator': 'exact'} for (property_key, property_value) in elements_as_action.items() if property_value is not None]}}).to_params()
            return f'{self._base_uri}api/person/funnel/correlation/?{urllib.parse.urlencode(params)}&cache_invalidation_key={cache_invalidation_key}'
        (event_name, property_name, property_value) = event_definition['event'].split('::')
        params = self._filter.shallow_clone({'funnel_correlation_person_converted': 'true' if success else 'false', 'funnel_correlation_person_entity': {'id': event_name, 'type': 'events', 'properties': [{'key': property_name, 'value': property_value, 'type': 'event', 'operator': 'exact'}]}}).to_params()
        return f'{self._base_uri}api/person/funnel/correlation/?{urllib.parse.urlencode(params)}'

    def construct_person_properties_people_url(self, success: bool, event_definition: EventDefinition, cache_invalidation_key: str) -> str:
        if False:
            return 10
        (property_name, property_value) = event_definition['event'].split('::')
        prop_type = 'group' if self._filter.aggregation_group_type_index else 'person'
        params = self._filter.shallow_clone({'funnel_correlation_person_converted': 'true' if success else 'false', 'funnel_correlation_property_values': [{'key': property_name, 'value': property_value, 'type': prop_type, 'operator': 'exact', 'group_type_index': self._filter.aggregation_group_type_index}]}).to_params()
        return f'{self._base_uri}api/person/funnel/correlation?{urllib.parse.urlencode(params)}&cache_invalidation_key={cache_invalidation_key}'

    def format_results(self, results: Tuple[List[EventOddsRatio], bool]) -> FunnelCorrelationResponse:
        if False:
            for i in range(10):
                print('nop')
        (odds_ratios, skewed_totals) = results
        return {'events': [self.serialize_event_odds_ratio(odds_ratio=odds_ratio) for odds_ratio in odds_ratios], 'skewed': skewed_totals}

    def run(self) -> FunnelCorrelationResponse:
        if False:
            i = 10
            return i + 15
        if not self._filter.entities:
            return FunnelCorrelationResponse(events=[], skewed=False)
        return self.format_results(self._run())

    def get_partial_event_contingency_tables(self) -> Tuple[List[EventContingencyTable], int, int]:
        if False:
            while True:
                i = 10
        "\n        For each event a person that started going through the funnel, gets stats\n        for how many of these users are sucessful and how many are unsuccessful.\n\n        It's a partial table as it doesn't include numbers of the negation of the\n        event, but does include the total success/failure numbers, which is enough\n        for us to calculate the odds ratio.\n        "
        (query, params) = self.get_contingency_table_query()
        results_with_total = insight_sync_execute(query, {**params, **self._filter.hogql_context.values}, query_type='funnel_correlation', filter=self._filter, team_id=self._team.pk)
        results = [result for result in results_with_total if result[0] != self.TOTAL_IDENTIFIER]
        (_, success_total, failure_total) = [result for result in results_with_total if result[0] == self.TOTAL_IDENTIFIER][0]
        return ([EventContingencyTable(event=result[0], visited=EventStats(success_count=result[1], failure_count=result[2]), success_total=success_total, failure_total=failure_total) for result in results], success_total, failure_total)

    def get_funnel_actors_cte(self) -> Tuple[str, Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        extra_fields = ['steps', 'final_timestamp', 'first_timestamp']
        for prop in self.properties_to_include:
            extra_fields.append(prop)
        return self._funnel_actors_generator.actor_query(limit_actors=False, extra_fields=extra_fields)

    @staticmethod
    def are_results_insignificant(event_contingency_table: EventContingencyTable) -> bool:
        if False:
            while True:
                i = 10
        '\n        Check if the results are insignificant, i.e. if the success/failure counts are\n        significantly different from the total counts\n        '
        total_count = event_contingency_table.success_total + event_contingency_table.failure_total
        if event_contingency_table.visited.success_count + event_contingency_table.visited.failure_count < min(FunnelCorrelation.MIN_PERSON_COUNT, FunnelCorrelation.MIN_PERSON_PERCENTAGE * total_count):
            return True
        return False

    def serialize_event_odds_ratio(self, odds_ratio: EventOddsRatio) -> EventOddsRatioSerialized:
        if False:
            return 10
        event_definition = self.serialize_event_with_property(event=odds_ratio['event'])
        cache_invalidation_key = generate_short_id()
        return {'success_count': odds_ratio['success_count'], 'success_people_url': self.construct_people_url(success=True, event_definition=event_definition, cache_invalidation_key=cache_invalidation_key), 'failure_count': odds_ratio['failure_count'], 'failure_people_url': self.construct_people_url(success=False, event_definition=event_definition, cache_invalidation_key=cache_invalidation_key), 'odds_ratio': odds_ratio['odds_ratio'], 'correlation_type': odds_ratio['correlation_type'], 'event': event_definition}

    def serialize_event_with_property(self, event: str) -> EventDefinition:
        if False:
            return 10
        '\n        Format the event name for display.\n        '
        if not self.support_autocapture_elements():
            return EventDefinition(event=event, properties={}, elements=[])
        (event_name, property_name, property_value) = event.split('::')
        if event_name == AUTOCAPTURE_EVENT and property_name == 'elements_chain':
            (event_type, elements_chain) = property_value.split(self.ELEMENTS_DIVIDER)
            return EventDefinition(event=event, properties={self.AUTOCAPTURE_EVENT_TYPE: event_type}, elements=cast(list, ElementSerializer(chain_to_elements(elements_chain), many=True).data))
        return EventDefinition(event=event, properties={}, elements=[])

def get_entity_odds_ratio(event_contingency_table: EventContingencyTable, prior_counts: int) -> EventOddsRatio:
    if False:
        while True:
            i = 10
    odds_ratio = (event_contingency_table.visited.success_count + prior_counts) * (event_contingency_table.failure_total - event_contingency_table.visited.failure_count + prior_counts) / ((event_contingency_table.success_total - event_contingency_table.visited.success_count + prior_counts) * (event_contingency_table.visited.failure_count + prior_counts))
    return EventOddsRatio(event=event_contingency_table.event, success_count=event_contingency_table.visited.success_count, failure_count=event_contingency_table.visited.failure_count, odds_ratio=odds_ratio, correlation_type='success' if odds_ratio > 1 else 'failure')

def build_selector(elements: List[Dict[str, Any]]) -> str:
    if False:
        return 10

    def element_to_selector(element: Dict[str, Any]) -> str:
        if False:
            for i in range(10):
                print('nop')
        if (attr_id := element.get('attr_id')):
            return f'[id="{attr_id}"]'
        return element['tag_name']
    return ' > '.join([element_to_selector(element) for element in elements])