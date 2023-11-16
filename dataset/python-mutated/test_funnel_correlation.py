import unittest
from rest_framework.exceptions import ValidationError
from ee.clickhouse.queries.funnels.funnel_correlation import EventContingencyTable, EventStats, FunnelCorrelation
from ee.clickhouse.queries.funnels.funnel_correlation_persons import FunnelCorrelationActors
from posthog.constants import INSIGHT_FUNNELS
from posthog.models.action import Action
from posthog.models.action_step import ActionStep
from posthog.models.element import Element
from posthog.models.filters import Filter
from posthog.models.group.util import create_group
from posthog.models.group_type_mapping import GroupTypeMapping
from posthog.models.instance_setting import override_instance_config
from posthog.test.base import APIBaseTest, ClickhouseTestMixin, _create_event, _create_person, also_test_with_materialized_columns, flush_persons_and_events, snapshot_clickhouse_queries, also_test_with_person_on_events_v2
from posthog.test.test_journeys import journeys_for

def _create_action(**kwargs):
    if False:
        print('Hello World!')
    team = kwargs.pop('team')
    name = kwargs.pop('name')
    properties = kwargs.pop('properties', {})
    action = Action.objects.create(team=team, name=name)
    ActionStep.objects.create(action=action, event=name, properties=properties)
    return action

class TestClickhouseFunnelCorrelation(ClickhouseTestMixin, APIBaseTest):
    maxDiff = None

    def _get_actors_for_event(self, filter: Filter, event_name: str, properties=None, success=True):
        if False:
            print('Hello World!')
        actor_filter = filter.shallow_clone({'funnel_correlation_person_entity': {'id': event_name, 'type': 'events', 'properties': properties}, 'funnel_correlation_person_converted': 'TrUe' if success else 'falSE'})
        (_, serialized_actors, _) = FunnelCorrelationActors(actor_filter, self.team).get_actors()
        return [str(row['id']) for row in serialized_actors]

    def _get_actors_for_property(self, filter: Filter, property_values: list, success=True):
        if False:
            for i in range(10):
                print('nop')
        actor_filter = filter.shallow_clone({'funnel_correlation_property_values': [{'key': prop, 'value': value, 'type': type, 'group_type_index': group_type_index} for (prop, value, type, group_type_index) in property_values], 'funnel_correlation_person_converted': 'TrUe' if success else 'falSE'})
        (_, serialized_actors, _) = FunnelCorrelationActors(actor_filter, self.team).get_actors()
        return [str(row['id']) for row in serialized_actors]

    def test_basic_funnel_correlation_with_events(self):
        if False:
            print('Hello World!')
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'events'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(10):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(10, 20):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [11, 1 / 11]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positively_related', 'success_count': 5, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'negatively_related', 'success_count': 0, 'failure_count': 5, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related')), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', success=False)), 0)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related', success=False)), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related')), 0)
        filter = filter.shallow_clone({'funnel_correlation_exclude_event_names': ['positively_related']})
        correlation = FunnelCorrelation(filter, self.team)
        result = correlation._run()[0]
        odds_ratio = result[0].pop('odds_ratio')
        expected_odds_ratio = 1 / 11
        self.assertAlmostEqual(odds_ratio, expected_odds_ratio)
        self.assertEqual(result, [{'event': 'negatively_related', 'success_count': 0, 'failure_count': 5, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related')), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', success=False)), 0)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related', success=False)), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related')), 0)

    @snapshot_clickhouse_queries
    def test_action_events_are_excluded_from_correlations(self):
        if False:
            i = 10
            return i + 15
        journey = {}
        for i in range(3):
            person_id = f'user_{i}'
            events = [{'event': 'user signed up', 'timestamp': '2020-01-02T14:00:00', 'properties': {'key': 'val'}}, {'event': 'user signed up', 'timestamp': '2020-01-02T14:10:00'}]
            if i % 2 == 0:
                events.append({'event': 'positively_related', 'timestamp': '2020-01-03T14:00:00'})
            events.append({'event': 'paid', 'timestamp': '2020-01-04T14:00:00', 'properties': {'key': 'val'}})
            journey[person_id] = events
        journey['failure'] = [{'event': 'user signed up', 'timestamp': '2020-01-02T14:00:00', 'properties': {'key': 'val'}}]
        journeys_for(events_by_person=journey, team=self.team)
        sign_up_action = _create_action(name='user signed up', team=self.team, properties=[{'key': 'key', 'type': 'event', 'value': ['val'], 'operator': 'exact'}])
        paid_action = _create_action(name='paid', team=self.team, properties=[{'key': 'key', 'type': 'event', 'value': ['val'], 'operator': 'exact'}])
        filters = {'events': [], 'actions': [{'id': sign_up_action.id, 'order': 0}, {'id': paid_action.id, 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'events'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        result = correlation._run()[0]
        self.assertEqual(result, [{'event': 'positively_related', 'success_count': 2, 'failure_count': 0, 'odds_ratio': 3, 'correlation_type': 'success'}])

    @also_test_with_person_on_events_v2
    @snapshot_clickhouse_queries
    def test_funnel_correlation_with_events_and_groups(self):
        if False:
            print('Hello World!')
        GroupTypeMapping.objects.create(team=self.team, group_type='organization', group_type_index=0)
        create_group(team_id=self.team.pk, group_type_index=0, group_key='org:5', properties={'industry': 'finance'})
        create_group(team_id=self.team.pk, group_type_index=0, group_key='org:7', properties={'industry': 'finance'})
        for i in range(10, 20):
            create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:{i}', properties={})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:{i}'})
            if i % 2 == 0:
                _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'$group_0': f'org:{i}'})
                _create_event(team=self.team, event='positively_related_without_group', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:{i}'})
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:5'})
        _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'$group_0': f'org:5'})
        _create_person(distinct_ids=[f'user_succ'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_succ', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:7'})
        _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'$group_0': f'org:7'})
        _create_event(team=self.team, event='paid', distinct_id=f'user_succ', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:7'})
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'events', 'aggregation_group_type_index': 0}
        filter = Filter(data=filters)
        result = FunnelCorrelation(filter, self.team)._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [12 / 7, 1 / 11]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positively_related', 'success_count': 5, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'negatively_related', 'success_count': 1, 'failure_count': 1, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related')), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', success=False)), 0)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related')), 1)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related', success=False)), 1)
        filter = filter.shallow_clone({'properties': [{'key': 'industry', 'value': 'finance', 'type': 'group', 'group_type_index': 0}]})
        result = FunnelCorrelation(filter, self.team)._run()[0]
        odds_ratio = result[0].pop('odds_ratio')
        expected_odds_ratio = 1
        self.assertAlmostEqual(odds_ratio, expected_odds_ratio)
        self.assertEqual(result, [{'event': 'negatively_related', 'success_count': 1, 'failure_count': 1, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related')), 1)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related', success=False)), 1)

    @also_test_with_materialized_columns(event_properties=[], person_properties=['$browser'])
    @snapshot_clickhouse_queries
    def test_basic_funnel_correlation_with_properties(self):
        if False:
            print('Hello World!')
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'properties', 'funnel_correlation_names': ['$browser']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(10):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(10, 20):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Negative'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk, properties={'$browser': 'Positive'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z')
        _create_person(distinct_ids=[f'user_succ'], team_id=self.team.pk, properties={'$browser': 'Negative'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_succ', timestamp='2020-01-02T14:00:00Z')
        _create_event(team=self.team, event='paid', distinct_id=f'user_succ', timestamp='2020-01-04T14:00:00Z')
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        prior_count = 1
        expected_odds_ratios = [(10 + prior_count) / (1 + prior_count) * ((11 - 1 + prior_count) / (11 - 10 + prior_count)), (1 + prior_count) / (10 + prior_count) * ((11 - 10 + prior_count) / (11 - 1 + prior_count))]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': '$browser::Positive', 'success_count': 10, 'failure_count': 1, 'correlation_type': 'success'}, {'event': '$browser::Negative', 'success_count': 1, 'failure_count': 10, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_property(filter, [('$browser', 'Positive', 'person', None)])), 10)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$browser', 'Positive', 'person', None)], False)), 1)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$browser', 'Negative', 'person', None)])), 1)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$browser', 'Negative', 'person', None)], False)), 10)

    @also_test_with_materialized_columns(event_properties=[], person_properties=['$browser'], verify_no_jsonextract=False)
    @snapshot_clickhouse_queries
    def test_funnel_correlation_with_properties_and_groups(self):
        if False:
            print('Hello World!')
        GroupTypeMapping.objects.create(team=self.team, group_type='organization', group_type_index=0)
        for i in range(10):
            create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:{i}', properties={'industry': 'positive'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:{i}'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:{i}'})
        for i in range(10, 20):
            create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:{i}', properties={'industry': 'negative'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Negative'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:{i}'})
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'$group_0': f'org:{i}'})
        create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:fail', properties={'industry': 'positive'})
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk, properties={'$browser': 'Positive'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:fail'})
        create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:succ', properties={'industry': 'negative'})
        _create_person(distinct_ids=[f'user_succ'], team_id=self.team.pk, properties={'$browser': 'Negative'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_succ', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:succ'})
        _create_event(team=self.team, event='paid', distinct_id=f'user_succ', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:succ'})
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'properties', 'funnel_correlation_names': ['industry'], 'aggregation_group_type_index': 0}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        prior_count = 1
        expected_odds_ratios = [(10 + prior_count) / (1 + prior_count) * ((11 - 1 + prior_count) / (11 - 10 + prior_count)), (1 + prior_count) / (10 + prior_count) * ((11 - 10 + prior_count) / (11 - 1 + prior_count))]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'industry::positive', 'success_count': 10, 'failure_count': 1, 'correlation_type': 'success'}, {'event': 'industry::negative', 'success_count': 1, 'failure_count': 10, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'positive', 'group', 0)])), 10)
        self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'positive', 'group', 0)], False)), 1)
        self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'negative', 'group', 0)])), 1)
        self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'negative', 'group', 0)], False)), 10)
        filter = filter.shallow_clone({'funnel_correlation_names': ['$all']})
        correlation = FunnelCorrelation(filter, self.team)
        new_result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in new_result]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(new_result, result)

    @also_test_with_materialized_columns(event_properties=[], person_properties=['$browser'], group_properties=[(0, 'industry')], verify_no_jsonextract=False)
    @also_test_with_person_on_events_v2
    @snapshot_clickhouse_queries
    def test_funnel_correlation_with_properties_and_groups_person_on_events(self):
        if False:
            i = 10
            return i + 15
        GroupTypeMapping.objects.create(team=self.team, group_type='organization', group_type_index=0)
        for i in range(10):
            create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:{i}', properties={'industry': 'positive'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:{i}'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:{i}'})
        for i in range(10, 20):
            create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:{i}', properties={'industry': 'negative'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Negative'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:{i}'})
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'$group_0': f'org:{i}'})
        create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:fail', properties={'industry': 'positive'})
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk, properties={'$browser': 'Positive'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:fail'})
        create_group(team_id=self.team.pk, group_type_index=0, group_key=f'org:succ', properties={'industry': 'negative'})
        _create_person(distinct_ids=[f'user_succ'], team_id=self.team.pk, properties={'$browser': 'Negative'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_succ', timestamp='2020-01-02T14:00:00Z', properties={'$group_0': f'org:succ'})
        _create_event(team=self.team, event='paid', distinct_id=f'user_succ', timestamp='2020-01-04T14:00:00Z', properties={'$group_0': f'org:succ'})
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'properties', 'funnel_correlation_names': ['industry'], 'aggregation_group_type_index': 0}
        with override_instance_config('PERSON_ON_EVENTS_ENABLED', True):
            filter = Filter(data=filters)
            correlation = FunnelCorrelation(filter, self.team)
            result = correlation._run()[0]
            odds_ratios = [item.pop('odds_ratio') for item in result]
            prior_count = 1
            expected_odds_ratios = [(10 + prior_count) / (1 + prior_count) * ((11 - 1 + prior_count) / (11 - 10 + prior_count)), (1 + prior_count) / (10 + prior_count) * ((11 - 10 + prior_count) / (11 - 1 + prior_count))]
            for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
                self.assertAlmostEqual(odds, expected_odds)
            self.assertEqual(result, [{'event': 'industry::positive', 'success_count': 10, 'failure_count': 1, 'correlation_type': 'success'}, {'event': 'industry::negative', 'success_count': 1, 'failure_count': 10, 'correlation_type': 'failure'}])
            self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'positive', 'group', 0)])), 10)
            self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'positive', 'group', 0)], False)), 1)
            self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'negative', 'group', 0)])), 1)
            self.assertEqual(len(self._get_actors_for_property(filter, [('industry', 'negative', 'group', 0)], False)), 10)
            filter = filter.shallow_clone({'funnel_correlation_names': ['$all']})
            correlation = FunnelCorrelation(filter, self.team)
            new_result = correlation._run()[0]
            odds_ratios = [item.pop('odds_ratio') for item in new_result]
            for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
                self.assertAlmostEqual(odds, expected_odds)
            self.assertEqual(new_result, result)

    def test_no_divide_by_zero_errors(self):
        if False:
            i = 10
            return i + 15
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(2):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='positive', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(2, 4):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Negative'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
        results = correlation._run()
        self.assertFalse(results[1])
        result = results[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [9, 1 / 3]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positive', 'success_count': 2, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'negatively_related', 'success_count': 0, 'failure_count': 1, 'correlation_type': 'failure'}])

    def test_correlation_with_properties_raises_validation_error(self):
        if False:
            i = 10
            return i + 15
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'properties'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        _create_person(distinct_ids=[f'user_1'], team_id=self.team.pk, properties={'$browser': 'Positive'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_1', timestamp='2020-01-02T14:00:00Z')
        _create_event(team=self.team, event='rick', distinct_id=f'user_1', timestamp='2020-01-03T14:00:00Z')
        _create_event(team=self.team, event='paid', distinct_id=f'user_1', timestamp='2020-01-04T14:00:00Z')
        flush_persons_and_events()
        with self.assertRaises(ValidationError):
            correlation._run()
        filter = filter.shallow_clone({'funnel_correlation_type': 'event_with_properties'})
        with self.assertRaises(ValidationError):
            FunnelCorrelation(filter, self.team)._run()

    @also_test_with_materialized_columns(event_properties=[], person_properties=['$browser'], verify_no_jsonextract=False)
    def test_correlation_with_multiple_properties(self):
        if False:
            i = 10
            return i + 15
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'properties', 'funnel_correlation_names': ['$browser', '$nice']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(5):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive', '$nice': 'very'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(5, 15):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Positive', '$nice': 'not'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(15, 20):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk, properties={'$browser': 'Negative', '$nice': 'smh'})
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk, properties={'$browser': 'Positive'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z')
        _create_person(distinct_ids=[f'user_succ'], team_id=self.team.pk, properties={'$browser': 'Negative'})
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_succ', timestamp='2020-01-02T14:00:00Z')
        _create_event(team=self.team, event='paid', distinct_id=f'user_succ', timestamp='2020-01-04T14:00:00Z')
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [16 / 2 * ((7 - 1) / (17 - 15)), 11 / 1 * ((7 - 0) / (17 - 10)), 6 / 1 * ((7 - 0) / (17 - 5)), 1 / 6 * ((7 - 5) / (17 - 0)), 2 / 6 * ((7 - 5) / (17 - 1)), 2 / 2 * ((7 - 1) / (17 - 1))]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        expected_result = [{'event': '$browser::Positive', 'success_count': 15, 'failure_count': 1, 'correlation_type': 'success'}, {'event': '$nice::not', 'success_count': 10, 'failure_count': 0, 'correlation_type': 'success'}, {'event': '$nice::very', 'success_count': 5, 'failure_count': 0, 'correlation_type': 'success'}, {'event': '$nice::smh', 'success_count': 0, 'failure_count': 5, 'correlation_type': 'failure'}, {'event': '$browser::Negative', 'success_count': 1, 'failure_count': 5, 'correlation_type': 'failure'}, {'event': '$nice::', 'success_count': 1, 'failure_count': 1, 'correlation_type': 'failure'}]
        self.assertEqual(result, expected_result)
        filter = filter.shallow_clone({'funnel_correlation_names': ['$all']})
        correlation = FunnelCorrelation(filter, self.team)
        new_result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in new_result]
        new_expected_odds_ratios = expected_odds_ratios[:-1]
        new_expected_result = expected_result[:-1]
        for (odds, expected_odds) in zip(odds_ratios, new_expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(new_result, new_expected_result)
        filter = filter.shallow_clone({'funnel_correlation_exclude_names': ['$browser']})
        correlation = FunnelCorrelation(filter, self.team)
        new_result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in new_result]
        new_expected_odds_ratios = expected_odds_ratios[1:4]
        new_expected_result = expected_result[1:4]
        for (odds, expected_odds) in zip(odds_ratios, new_expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(new_result, new_expected_result)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$nice', 'not', 'person', None)])), 10)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$nice', '', 'person', None)], False)), 1)
        self.assertEqual(len(self._get_actors_for_property(filter, [('$nice', 'very', 'person', None)])), 5)

    def test_discarding_insignificant_events(self):
        if False:
            while True:
                i = 10
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'events'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(10):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
            if i % 10 == 0:
                _create_event(team=self.team, event='low_sig_positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:20:00Z')
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(10, 20):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
            if i % 5 == 0:
                _create_event(team=self.team, event='low_sig_negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z')
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.11
        FunnelCorrelation.MIN_PERSON_COUNT = 25
        result = correlation._run()[0]
        self.assertEqual(len(result), 2)

    def test_events_within_conversion_window_for_correlation(self):
        if False:
            for i in range(10):
                print('nop')
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'funnel_window_interval': '10', 'funnel_window_interval_unit': 'minute', 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'events'}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        _create_person(distinct_ids=['user_successful'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id='user_successful', timestamp='2020-01-02T14:00:00Z')
        _create_event(team=self.team, event='positively_related', distinct_id='user_successful', timestamp='2020-01-02T14:02:00Z')
        _create_event(team=self.team, event='paid', distinct_id='user_successful', timestamp='2020-01-02T14:06:00Z')
        _create_person(distinct_ids=['user_dropoff'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id='user_dropoff', timestamp='2020-01-02T14:00:00Z')
        _create_event(team=self.team, event='NOT_negatively_related', distinct_id='user_dropoff', timestamp='2020-01-02T14:15:00Z')
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [4]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positively_related', 'success_count': 1, 'failure_count': 0, 'correlation_type': 'success'}])

    @also_test_with_materialized_columns(['blah', 'signup_source'], verify_no_jsonextract=False)
    def test_funnel_correlation_with_event_properties(self):
        if False:
            i = 10
            return i + 15
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'event_with_properties', 'funnel_correlation_event_names': ['positively_related', 'negatively_related']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(10):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'facebook' if i % 4 == 0 else 'email', 'blah': 'value_bleh'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        for i in range(10, 20):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'shazam' if i % 6 == 0 else 'email'})
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [11, 5.5, 2 / 11]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positively_related::blah::value_bleh', 'success_count': 5, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'positively_related::signup_source::facebook', 'success_count': 3, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'negatively_related::signup_source::email', 'success_count': 0, 'failure_count': 3, 'correlation_type': 'failure'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', {'blah': 'value_bleh'})), 5)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', {'signup_source': 'facebook'})), 3)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', {'signup_source': 'facebook'}, False)), 0)
        self.assertEqual(len(self._get_actors_for_event(filter, 'negatively_related', {'signup_source': 'email'}, False)), 3)

    @also_test_with_materialized_columns(['blah', 'signup_source'], verify_no_jsonextract=False)
    @snapshot_clickhouse_queries
    def test_funnel_correlation_with_event_properties_and_groups(self):
        if False:
            while True:
                i = 10
        GroupTypeMapping.objects.create(team=self.team, group_type='organization', group_type_index=1)
        for i in range(10):
            create_group(team_id=self.team.pk, group_type_index=1, group_key=f'org:{i}', properties={'industry': 'positive'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_1': f'org:{i}'})
            if i % 2 == 0:
                _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'facebook' if i % 4 == 0 else 'email', 'blah': 'value_bleh', '$group_1': f'org:{i}'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z', properties={'$group_1': f'org:{i}'})
        for i in range(10, 20):
            create_group(team_id=self.team.pk, group_type_index=1, group_key=f'org:{i}', properties={'industry': 'positive'})
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z', properties={'$group_1': f'org:{i}'})
            if i % 2 == 0:
                _create_event(team=self.team, event='negatively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'shazam' if i % 6 == 0 else 'email', '$group_1': f'org:{i}'})
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'aggregation_group_type_index': 1, 'funnel_correlation_type': 'event_with_properties', 'funnel_correlation_event_names': ['positively_related', 'negatively_related']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        result = correlation._run()[0]
        odds_ratios = [item.pop('odds_ratio') for item in result]
        expected_odds_ratios = [11, 5.5, 2 / 11]
        for (odds, expected_odds) in zip(odds_ratios, expected_odds_ratios):
            self.assertAlmostEqual(odds, expected_odds)
        self.assertEqual(result, [{'event': 'positively_related::blah::value_bleh', 'success_count': 5, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'positively_related::signup_source::facebook', 'success_count': 3, 'failure_count': 0, 'correlation_type': 'success'}, {'event': 'negatively_related::signup_source::email', 'success_count': 0, 'failure_count': 3, 'correlation_type': 'failure'}])

    def test_funnel_correlation_with_event_properties_exclusions(self):
        if False:
            return 10
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'event_with_properties', 'funnel_correlation_event_names': ['positively_related'], 'funnel_correlation_event_exclude_property_names': ['signup_source']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(3):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='positively_related', distinct_id=f'user_{i}', timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'facebook', 'blah': 'value_bleh'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z')
        result = correlation._run()[0]
        self.assertEqual(result, [{'event': 'positively_related::blah::value_bleh', 'success_count': 3, 'failure_count': 0, 'odds_ratio': 8, 'correlation_type': 'success'}])
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', {'blah': 'value_bleh'})), 3)
        self.assertEqual(len(self._get_actors_for_event(filter, 'positively_related', {'signup_source': 'facebook'})), 3)

    @also_test_with_materialized_columns(['$event_type', 'signup_source'])
    def test_funnel_correlation_with_event_properties_autocapture(self):
        if False:
            i = 10
            return i + 15
        filters = {'events': [{'id': 'user signed up', 'type': 'events', 'order': 0}, {'id': 'paid', 'type': 'events', 'order': 1}], 'insight': INSIGHT_FUNNELS, 'date_from': '2020-01-01', 'date_to': '2020-01-14', 'funnel_correlation_type': 'event_with_properties', 'funnel_correlation_event_names': ['$autocapture']}
        filter = Filter(data=filters)
        correlation = FunnelCorrelation(filter, self.team)
        for i in range(6):
            _create_person(distinct_ids=[f'user_{i}'], team_id=self.team.pk)
            _create_event(team=self.team, event='user signed up', distinct_id=f'user_{i}', timestamp='2020-01-02T14:00:00Z')
            _create_event(team=self.team, event='$autocapture', distinct_id=f'user_{i}', elements=[Element(nth_of_type=1, nth_child=0, tag_name='a', href='/movie')], timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'email', '$event_type': 'click'})
            if i % 2 == 0:
                _create_event(team=self.team, event='$autocapture', distinct_id=f'user_{i}', elements=[Element(nth_of_type=1, nth_child=0, tag_name='button', text='Pay $10')], timestamp='2020-01-03T14:00:00Z', properties={'signup_source': 'facebook', '$event_type': 'submit'})
            _create_event(team=self.team, event='paid', distinct_id=f'user_{i}', timestamp='2020-01-04T14:00:00Z')
        _create_person(distinct_ids=[f'user_fail'], team_id=self.team.pk)
        _create_event(team=self.team, event='user signed up', distinct_id=f'user_fail', timestamp='2020-01-02T14:00:00Z')
        result = correlation._run()[0]
        self.assertEqual(result, [{'event': '$autocapture::elements_chain::click__~~__a:href="/movie"nth-child="0"nth-of-type="1"', 'success_count': 6, 'failure_count': 0, 'odds_ratio': 14.0, 'correlation_type': 'success'}, {'event': '$autocapture::elements_chain::submit__~~__button:nth-child="0"nth-of-type="1"text="Pay $10"', 'success_count': 3, 'failure_count': 0, 'odds_ratio': 2.0, 'correlation_type': 'success'}])
        self.assertEqual(len(self._get_actors_for_event(filter, '$autocapture', {'signup_source': 'facebook'})), 3)
        self.assertEqual(len(self._get_actors_for_event(filter, '$autocapture', {'$event_type': 'click'})), 6)
        self.assertEqual(len(self._get_actors_for_event(filter, '$autocapture', [{'key': 'tag_name', 'operator': 'exact', 'type': 'element', 'value': 'button'}, {'key': 'text', 'operator': 'exact', 'type': 'element', 'value': 'Pay $10'}])), 3)
        self.assertEqual(len(self._get_actors_for_event(filter, '$autocapture', [{'key': 'tag_name', 'operator': 'exact', 'type': 'element', 'value': 'a'}, {'key': 'href', 'operator': 'exact', 'type': 'element', 'value': '/movie'}])), 6)

class TestCorrelationFunctions(unittest.TestCase):

    def test_are_results_insignificant(self):
        if False:
            return 10
        contingency_tables = [EventContingencyTable(event='negatively_related', visited=EventStats(success_count=0, failure_count=5), success_total=10, failure_total=10), EventContingencyTable(event='positively_related', visited=EventStats(success_count=5, failure_count=0), success_total=10, failure_total=10), EventContingencyTable(event='low_sig_negatively_related', visited=EventStats(success_count=0, failure_count=2), success_total=10, failure_total=10), EventContingencyTable(event='low_sig_positively_related', visited=EventStats(success_count=1, failure_count=0), success_total=10, failure_total=10)]
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.11
        FunnelCorrelation.MIN_PERSON_COUNT = 25
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 2)
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.051
        FunnelCorrelation.MIN_PERSON_COUNT = 25
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 3)
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.5
        FunnelCorrelation.MIN_PERSON_COUNT = 3
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 2)
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.5
        FunnelCorrelation.MIN_PERSON_COUNT = 2
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 3)
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.5
        FunnelCorrelation.MIN_PERSON_COUNT = 100
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 0)
        FunnelCorrelation.MIN_PERSON_PERCENTAGE = 0.5
        FunnelCorrelation.MIN_PERSON_COUNT = 6
        result = [1 for contingency_table in contingency_tables if not FunnelCorrelation.are_results_insignificant(contingency_table)]
        self.assertEqual(len(result), 0)