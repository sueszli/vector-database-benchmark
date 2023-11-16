import datetime
import pytest
import pytz
from .....product.models import Collection, CollectionChannelListing
from ....tests.utils import assert_graphql_error_with_message, get_graphql_content

@pytest.fixture
def collections_for_sorting_with_channels(channel_USD, channel_PLN):
    if False:
        print('Hello World!')
    collections = Collection.objects.bulk_create([Collection(name='Collection1', slug='collection1'), Collection(name='Collection2', slug='collection2'), Collection(name='Collection3', slug='collection3'), Collection(name='Collection4', slug='collection4'), Collection(name='Collection5', slug='collection5')])
    CollectionChannelListing.objects.bulk_create([CollectionChannelListing(collection=collections[0], published_at=None, is_published=True, channel=channel_USD), CollectionChannelListing(collection=collections[1], published_at=None, is_published=False, channel=channel_USD), CollectionChannelListing(collection=collections[2], published_at=datetime.datetime(2004, 1, 1, tzinfo=pytz.UTC), is_published=False, channel=channel_USD), CollectionChannelListing(collection=collections[3], published_at=datetime.datetime(2003, 1, 1, tzinfo=pytz.UTC), is_published=False, channel=channel_USD), CollectionChannelListing(collection=collections[0], published_at=None, is_published=False, channel=channel_PLN), CollectionChannelListing(collection=collections[1], published_at=None, is_published=True, channel=channel_PLN), CollectionChannelListing(collection=collections[2], published_at=datetime.datetime(2002, 1, 1, tzinfo=pytz.UTC), is_published=False, channel=channel_PLN), CollectionChannelListing(collection=collections[4], published_at=datetime.datetime(2001, 1, 1, tzinfo=pytz.UTC), is_published=False, channel=channel_PLN)])
    return collections
QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING = '\n    query (\n        $sortBy: CollectionSortingInput,\n        $filter: CollectionFilterInput, $channel: String\n    ){\n        collections (\n            first: 10, sortBy: $sortBy, filter: $filter, channel: $channel\n        ) {\n            edges {\n                node {\n                    name\n                    slug\n                }\n            }\n        }\n    }\n'

@pytest.mark.parametrize('sort_by', [{'field': 'AVAILABILITY', 'direction': 'ASC'}, {'field': 'PUBLICATION_DATE', 'direction': 'DESC'}])
def test_collections_with_sorting_and_without_channel(sort_by, staff_api_client, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    variables = {'sortBy': sort_by}
    response = staff_api_client.post_graphql(QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING, variables, permissions=[permission_manage_products], check_no_permissions=False)
    assert_graphql_error_with_message(response, 'A default channel does not exist.')

@pytest.mark.parametrize(('sort_by', 'collections_order'), [({'field': 'PUBLICATION_DATE', 'direction': 'ASC'}, ['Collection4', 'Collection3', 'Collection1', 'Collection2']), ({'field': 'PUBLICATION_DATE', 'direction': 'DESC'}, ['Collection2', 'Collection1', 'Collection3', 'Collection4'])])
def test_collections_with_sorting_and_channel_USD(sort_by, collections_order, staff_api_client, permission_manage_products, collections_for_sorting_with_channels, channel_USD):
    if False:
        print('Hello World!')
    variables = {'sortBy': sort_by, 'channel': channel_USD.slug}
    response = staff_api_client.post_graphql(QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    collections_nodes = content['data']['collections']['edges']
    for (index, collection_name) in enumerate(collections_order):
        assert collection_name == collections_nodes[index]['node']['name']

@pytest.mark.parametrize(('sort_by', 'collections_order'), [({'field': 'PUBLICATION_DATE', 'direction': 'ASC'}, ['Collection5', 'Collection3', 'Collection1', 'Collection2']), ({'field': 'PUBLICATION_DATE', 'direction': 'DESC'}, ['Collection2', 'Collection1', 'Collection3', 'Collection5'])])
def test_collections_with_sorting_and_channel_PLN(sort_by, collections_order, staff_api_client, permission_manage_products, collections_for_sorting_with_channels, channel_PLN):
    if False:
        i = 10
        return i + 15
    variables = {'sortBy': sort_by, 'channel': channel_PLN.slug}
    response = staff_api_client.post_graphql(QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    collections_nodes = content['data']['collections']['edges']
    for (index, collection_name) in enumerate(collections_order):
        assert collection_name == collections_nodes[index]['node']['name']

@pytest.mark.parametrize('sort_by', [{'field': 'AVAILABILITY', 'direction': 'ASC'}, {'field': 'PUBLICATION_DATE', 'direction': 'ASC'}])
def test_collections_with_sorting_and_not_existing_channel_asc(sort_by, staff_api_client, permission_manage_products, collections_for_sorting_with_channels, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    variables = {'sortBy': sort_by, 'channel': 'Not-existing'}
    response = staff_api_client.post_graphql(QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    assert not content['data']['collections']['edges']

@pytest.mark.parametrize('sort_by', [{'field': 'AVAILABILITY', 'direction': 'DESC'}, {'field': 'PUBLICATION_DATE', 'direction': 'DESC'}])
def test_collections_with_sorting_and_not_existing_channel_desc(sort_by, staff_api_client, permission_manage_products, collections_for_sorting_with_channels, channel_USD):
    if False:
        while True:
            i = 10
    variables = {'sortBy': sort_by, 'channel': 'Not-existing'}
    response = staff_api_client.post_graphql(QUERY_COLLECTIONS_WITH_SORTING_AND_FILTERING, variables, permissions=[permission_manage_products], check_no_permissions=False)
    content = get_graphql_content(response)
    assert not content['data']['collections']['edges']