from ee.clickhouse.queries.groups_join_query import GroupsJoinQuery
from posthog.models.filters import Filter

def test_groups_join_query_blank():
    if False:
        return 10
    filter = Filter(data={'properties': []})
    assert GroupsJoinQuery(filter, 2).get_join_query() == ('', {})

def test_groups_join_query_filtering(snapshot):
    if False:
        while True:
            i = 10
    filter = Filter(data={'properties': [{'key': 'industry', 'value': 'finance', 'type': 'group', 'group_type_index': 0}]})
    assert GroupsJoinQuery(filter, 2).get_join_query() == snapshot

def test_groups_join_query_filtering_with_custom_key_names(snapshot):
    if False:
        while True:
            i = 10
    filter = Filter(data={'properties': [{'key': 'industry', 'value': 'finance', 'type': 'group', 'group_type_index': 0}, {'key': 'company', 'value': 'crashed', 'type': 'group', 'group_type_index': 2}]})
    assert GroupsJoinQuery(filter, 2, join_key='call_me_industry').get_join_query() == snapshot