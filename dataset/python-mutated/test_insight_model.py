from django.db.utils import IntegrityError
from posthog.models import Dashboard, Insight, Team
from posthog.models.insight import generate_insight_cache_key
from posthog.test.base import BaseTest

class TestInsightModel(BaseTest):

    def test_short_id_and_team_must_be_unique_together(self) -> None:
        if False:
            while True:
                i = 10
        Insight.objects.create(team=self.team, short_id='123456')
        new_team = Team.objects.create(organization=self.organization)
        Insight.objects.create(team=new_team, short_id='123456')
        count = Insight.objects.count()
        with self.assertRaises(IntegrityError):
            Insight.objects.create(team=self.team, short_id='123456')
            self.assertEqual(Insight.objects.count(), count)

    def test_short_id_is_automatically_generated(self) -> None:
        if False:
            print('Hello World!')
        d = Insight.objects.create(team=self.team)
        self.assertRegex(d.short_id, '[0-9A-Za-z_-]{8}')

    def test_dashboard_with_no_filters_does_not_override_date_from(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        filters_with_no_dashboard = insight.dashboard_filters(dashboard=None)
        filters_with_dashboard_with_no_date_from = insight.dashboard_filters(dashboard=Dashboard.objects.create(team=self.team))
        assert filters_with_no_dashboard['date_from'] == '-30d'
        assert filters_with_dashboard_with_no_date_from['date_to'] is None
        del filters_with_dashboard_with_no_date_from['date_to']
        assert filters_with_no_dashboard == filters_with_dashboard_with_no_date_from

    def test_dashboard_with_date_from_filters_does_override_date_from(self) -> None:
        if False:
            while True:
                i = 10
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        filters_with_dashboard_with_different_date_from = insight.dashboard_filters(dashboard=Dashboard.objects.create(team=self.team, filters={'date_from': '-14d'}))
        assert filters_with_dashboard_with_different_date_from['date_from'] == '-14d'

    def test_dashboard_with_date_from_filters_does_override_date_to(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight = Insight.objects.create(team=self.team, filters={'date_from': '2023-06-17', 'date_to': '2023-06-25'})
        filters_with_dashboard_with_different_date_from = insight.dashboard_filters(dashboard=Dashboard.objects.create(team=self.team, filters={'date_from': '-14d'}))
        assert filters_with_dashboard_with_different_date_from['date_from'] == '-14d'
        assert filters_with_dashboard_with_different_date_from['date_to'] is None

    def test_dashboard_with_same_date_from_filters_generates_expected_date_from(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        filters_with_dashboard_with_same_date_from = insight.dashboard_filters(dashboard=Dashboard.objects.create(team=self.team, filters={'date_from': '-30d'}))
        assert filters_with_dashboard_with_same_date_from['date_from'] == '-30d'

    def test_dashboard_with_date_from_all_overrides_compare(self) -> None:
        if False:
            i = 10
            return i + 15
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d', 'compare': True})
        dashboard = Dashboard.objects.create(team=self.team, filters={'date_from': 'all'})
        filters = insight.dashboard_filters(dashboard=dashboard)
        assert filters['compare'] is None

    def test_dashboard_does_not_affect_filters_hash_with_absent_date_from(self) -> None:
        if False:
            while True:
                i = 10
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        dashboard = Dashboard.objects.create(team=self.team, filters={})
        filters_hash_no_dashboard = generate_insight_cache_key(insight, None)
        filters_hash_with_absent_date_from = generate_insight_cache_key(insight, dashboard)
        assert filters_hash_no_dashboard == filters_hash_with_absent_date_from

    def test_dashboard_does_not_affect_filters_hash_with_null_date_from(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        dashboard = Dashboard.objects.create(team=self.team, filters={'date_from': None})
        filters_hash_no_dashboard = generate_insight_cache_key(insight, None)
        filters_hash_with_null_date_from = generate_insight_cache_key(insight, dashboard)
        assert filters_hash_no_dashboard == filters_hash_with_null_date_from

    def test_dashboard_with_date_from_changes_filters_hash(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight = Insight.objects.create(team=self.team, filters={'date_from': '-30d'})
        dashboard = Dashboard.objects.create(team=self.team, filters={'date_from': '-90d'})
        filters_hash_one = generate_insight_cache_key(insight, None)
        filters_hash_two = generate_insight_cache_key(insight, dashboard)
        assert filters_hash_one != filters_hash_two

    def test_query_hash_matches_same_query_source(self) -> None:
        if False:
            print('Hello World!')
        insight_with_query_at_top_level = Insight.objects.create(team=self.team, query={'kind': 'EventsQuery'})
        insight_with_query_in_source = Insight.objects.create(team=self.team, query={'kind': 'DataTable', 'source': {'kind': 'EventsQuery'}})
        filters_hash_one = generate_insight_cache_key(insight_with_query_at_top_level, None)
        filters_hash_two = generate_insight_cache_key(insight_with_query_in_source, None)
        assert filters_hash_one == filters_hash_two

    def test_query_hash_varies_with_query_content(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        insight_one = Insight.objects.create(team=self.team, query={'kind': 'EventsQuery'})
        insight_two = Insight.objects.create(team=self.team, query={'kind': 'EventsQuery', 'anything': 'else'})
        filters_hash_one = generate_insight_cache_key(insight_one, None)
        filters_hash_two = generate_insight_cache_key(insight_two, None)
        assert filters_hash_one != filters_hash_two

    def test_dashboard_with_query_insight_and_filters(self) -> None:
        if False:
            i = 10
            return i + 15
        browser_equals_firefox = {'key': '$browser', 'label': None, 'operator': 'exact', 'type': 'event', 'value': ['Firefox']}
        browser_equals_chrome = {'key': '$browser', 'label': None, 'operator': 'exact', 'type': 'event', 'value': ['Chrome']}
        test_cases = [({'dateRange': {'date_from': '-14d', 'date_to': '-7d'}}, {}, {'dateRange': {'date_from': '-14d', 'date_to': '-7d'}, 'properties': None}), ({}, {'date_from': '-14d', 'date_to': '-7d'}, {'dateRange': {'date_from': '-14d', 'date_to': '-7d'}, 'properties': None}), ({'dateRange': {'date_from': '-2d', 'date_to': '-1d'}}, {'date_from': '-4d', 'date_to': '-3d'}, {'dateRange': {'date_from': '-4d', 'date_to': '-3d'}, 'properties': None}), ({'dateRange': {'date_from': '-14d', 'date_to': '-7d'}}, {'date_from': 'all'}, {'dateRange': {'date_from': 'all', 'date_to': None}, 'properties': None}), ({}, {}, {'dateRange': {'date_from': None, 'date_to': None}, 'properties': None}), ({'properties': [browser_equals_firefox]}, {}, {'dateRange': {'date_from': None, 'date_to': None}, 'properties': [browser_equals_firefox]}), ({}, {'properties': [browser_equals_chrome]}, {'dateRange': {'date_from': None, 'date_to': None}, 'properties': [browser_equals_chrome]}), ({'properties': [browser_equals_firefox]}, {'properties': [browser_equals_chrome]}, {'dateRange': {'date_from': None, 'date_to': None}, 'properties': [browser_equals_firefox, browser_equals_chrome]})]
        for (query_filters, dashboard_filters, expected_filters) in test_cases:
            query_insight = Insight.objects.create(team=self.team, query={'kind': 'DataTableNode', 'source': {'filters': query_filters, 'kind': 'HogQLQuery', 'modifiers': None, 'query': 'select * from events where {filters}', 'response': None, 'values': None}})
            dashboard = Dashboard.objects.create(team=self.team, filters=dashboard_filters)
            data = query_insight.dashboard_query(dashboard)
            assert data
            actual = data['source']['filters']
            assert expected_filters == actual

    def test_query_hash_varies_with_dashboard_filters(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        query = {'kind': 'DataTableNode', 'source': {'filters': {'dateRange': {'date_from': '-14d', 'date_to': '-7d'}}, 'kind': 'HogQLQuery', 'modifiers': None, 'query': 'select * from events where {filters}', 'response': None, 'values': None}}
        dashboard_filters = {'date_from': '-4d', 'date_to': '-3d'}
        query_insight = Insight.objects.create(team=self.team, query=query)
        dashboard = Dashboard.objects.create(team=self.team, filters=dashboard_filters)
        hash_sans_dashboard = generate_insight_cache_key(query_insight, None)
        hash_with_dashboard = generate_insight_cache_key(query_insight, dashboard)
        assert hash_sans_dashboard != hash_with_dashboard