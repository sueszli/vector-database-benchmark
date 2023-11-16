import graphene
import pytest
from .....tax.models import TaxClass
from ....tests.utils import assert_no_permission, get_graphql_content
from ..fragments import TAX_CLASS_FRAGMENT
QUERY = '\n    query TaxClasses($sortBy: TaxClassSortingInput, $filter: TaxClassFilterInput) {\n        taxClasses(first: 100, sortBy: $sortBy, filter: $filter) {\n            totalCount\n            edges {\n                node {\n                    ...TaxClass\n                }\n            }\n        }\n    }\n    ' + TAX_CLASS_FRAGMENT

def test_tax_classes_query_no_permissions(user_api_client):
    if False:
        return 10
    response = user_api_client.post_graphql(QUERY, {}, permissions=[])
    assert_no_permission(response)

def test_tax_classes_query_staff_user(staff_api_client):
    if False:
        print('Hello World!')
    total_count = TaxClass.objects.count()
    response = staff_api_client.post_graphql(QUERY, {})
    content = get_graphql_content(response)
    edges = content['data']['taxClasses']['edges']
    assert content['data']['taxClasses']['totalCount'] == total_count
    assert len(edges) == total_count
    assert edges[0]['node']

def test_tax_classes_query_app(app_api_client):
    if False:
        while True:
            i = 10
    total_count = TaxClass.objects.count()
    response = app_api_client.post_graphql(QUERY, {})
    content = get_graphql_content(response)
    edges = content['data']['taxClasses']['edges']
    assert content['data']['taxClasses']['totalCount'] == total_count
    assert len(edges) == total_count
    assert edges[0]['node']

def test_tax_classes_filter_by_ids(staff_api_client):
    if False:
        return 10
    id = graphene.Node.to_global_id('TaxClass', TaxClass.objects.first().pk)
    ids = [id]
    response = staff_api_client.post_graphql(QUERY, {'filter': {'ids': ids}})
    content = get_graphql_content(response)
    edges = content['data']['taxClasses']['edges']
    assert len(edges) == 1
    assert edges[0]['node']['id'] == id

@pytest.mark.parametrize(('country', 'count'), [('PL', 1), ('US', 0)])
def test_tax_classes_filter_by_countries(country, count, staff_api_client):
    if False:
        while True:
            i = 10
    filter = {'filter': {'countries': [country]}}
    response = staff_api_client.post_graphql(QUERY, filter)
    content = get_graphql_content(response)
    edges = content['data']['taxClasses']['edges']
    assert len(edges) == count