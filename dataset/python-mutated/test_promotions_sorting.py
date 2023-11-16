from datetime import timedelta
import pytest
from django.utils import timezone
from ....tests.utils import get_graphql_content
QUERY_PROMOTIONS = '\n    query Promotions($sortBy: PromotionSortingInput){\n        promotions(first: 10, sortBy: $sortBy) {\n            edges {\n                node {\n                    id\n                    name\n                    description\n                    startDate\n                    rules {\n                        id\n                    }\n                }\n            }\n        }\n    }\n'

@pytest.mark.parametrize('direction', ['ASC', 'DESC'])
def test_sorting_promotions_by_name(direction, staff_api_client, promotion, promotion_list, permission_manage_discounts):
    if False:
        return 10
    promotion_list.insert(0, promotion)
    variables = {'sortBy': {'direction': direction, 'field': 'NAME'}}
    response = staff_api_client.post_graphql(QUERY_PROMOTIONS, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    promotions = content['data']['promotions']['edges']
    if direction == 'DESC':
        promotions.reverse()
    assert len(promotions) == 4
    assert [promotion['node']['name'] for promotion in promotions] == [promotion.name for promotion in promotion_list]

@pytest.mark.parametrize('direction', ['ASC', 'DESC'])
def test_sorting_promotions_by_end_date(direction, staff_api_client, promotion, promotion_list, permission_manage_discounts):
    if False:
        while True:
            i = 10
    promotion_list.insert(2, promotion)
    variables = {'sortBy': {'direction': direction, 'field': 'END_DATE'}}
    response = staff_api_client.post_graphql(QUERY_PROMOTIONS, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    promotions = content['data']['promotions']['edges']
    if direction == 'DESC':
        promotions.reverse()
    assert len(promotions) == 4
    assert [promotion['node']['name'] for promotion in promotions] == [promotion.name for promotion in promotion_list]

@pytest.mark.parametrize('direction', ['ASC', 'DESC'])
def test_sorting_promotions_by_start_date(direction, staff_api_client, promotion, promotion_list, permission_manage_discounts):
    if False:
        i = 10
        return i + 15
    promotion.start_date = timezone.now() + timedelta(days=3)
    promotion.save(update_fields=['start_date'])
    promotion_list.insert(1, promotion)
    variables = {'sortBy': {'direction': direction, 'field': 'START_DATE'}}
    response = staff_api_client.post_graphql(QUERY_PROMOTIONS, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    promotions = content['data']['promotions']['edges']
    if direction == 'DESC':
        promotions.reverse()
    assert len(promotions) == 4
    assert [promotion['node']['name'] for promotion in promotions] == [promotion.name for promotion in promotion_list]

@pytest.mark.parametrize('direction', ['ASC', 'DESC'])
def test_sorting_promotions_by_created_at(direction, staff_api_client, promotion_list, permission_manage_discounts):
    if False:
        return 10
    variables = {'sortBy': {'direction': direction, 'field': 'CREATED_AT'}}
    response = staff_api_client.post_graphql(QUERY_PROMOTIONS, variables, permissions=[permission_manage_discounts])
    content = get_graphql_content(response)
    promotions = content['data']['promotions']['edges']
    if direction == 'DESC':
        promotions.reverse()
    assert len(promotions) == 3
    assert [promotion['node']['name'] for promotion in promotions] == [promotion.name for promotion in promotion_list]