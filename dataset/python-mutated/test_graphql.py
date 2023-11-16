from functools import partial
from unittest import mock
from unittest.mock import Mock
import graphene
import pytest
from django.urls import reverse
from graphql.error import GraphQLError
from graphql_relay import to_global_id
from ....order import models as order_models
from ...core.utils import from_global_id_or_error
from ...order.types import Order
from ...product.types import Product
from ...tests.utils import get_graphql_content
from ...utils import get_nodes

def test_middleware_dont_generate_sql_requests(client, settings, assert_num_queries):
    if False:
        for i in range(10):
            print('nop')
    'Test that a GET request results in no database queries.'
    settings.DEBUG = True
    with assert_num_queries(0):
        response = client.get(reverse('api'))
        assert response.status_code == 200

def test_jwt_middleware(client, admin_user):
    if False:
        for i in range(10):
            print('nop')
    user_details_query = '\n        {\n          me {\n            email\n          }\n        }\n    '
    create_token_query = '\n        mutation {\n          tokenCreate(email: "admin@example.com", password: "password") {\n            token\n          }\n        }\n    '
    api_url = reverse('api')
    api_client_post = partial(client.post, api_url, content_type='application/json')
    response = api_client_post(data={'query': user_details_query})
    repl_data = response.json()
    assert response.status_code == 200
    assert not response.wsgi_request.user
    assert repl_data['data']['me'] is None
    response = api_client_post(data={'query': create_token_query})
    repl_data = response.json()
    assert response.status_code == 200
    assert response.wsgi_request.user == admin_user
    token = repl_data['data']['tokenCreate']['token']
    assert token is not None
    response = api_client_post(data={'query': user_details_query}, HTTP_AUTHORIZATION=f'JWT {token}')
    repl_data = response.json()
    assert response.status_code == 200
    assert response.wsgi_request.user == admin_user
    assert 'errors' not in repl_data
    assert repl_data['data']['me'] == {'email': admin_user.email}

def test_real_query(user_api_client, product, channel_USD):
    if False:
        for i in range(10):
            print('nop')
    product_attr = product.product_type.product_attributes.first()
    category = product.category
    attr_value = product_attr.values.first()
    query = '\n    query Root($categoryId: ID!, $sortBy: ProductOrder, $first: Int,\n            $attributesFilter: [AttributeInput!], $channel: String) {\n\n        category(id: $categoryId) {\n            ...CategoryPageFragmentQuery\n            __typename\n        }\n        products(first: $first, sortBy: $sortBy, filter: {categories: [$categoryId],\n            attributes: $attributesFilter}, channel: $channel) {\n\n            ...ProductListFragmentQuery\n            __typename\n        }\n        attributes(first: 20, filter: {inCategory: $categoryId}, channel: $channel) {\n            edges {\n                node {\n                    ...ProductFiltersFragmentQuery\n                    __typename\n                }\n            }\n        }\n    }\n\n    fragment CategoryPageFragmentQuery on Category {\n        id\n        name\n        ancestors(first: 20) {\n            edges {\n                node {\n                    name\n                    id\n                    __typename\n                }\n            }\n        }\n        children(first: 20) {\n            edges {\n                node {\n                    name\n                    id\n                    slug\n                    __typename\n                }\n            }\n        }\n        __typename\n    }\n\n    fragment ProductListFragmentQuery on ProductCountableConnection {\n        edges {\n            node {\n                ...ProductFragmentQuery\n                __typename\n            }\n            __typename\n        }\n        pageInfo {\n            hasNextPage\n            __typename\n        }\n        __typename\n    }\n\n    fragment ProductFragmentQuery on Product {\n        id\n        isAvailable\n        name\n        pricing {\n            ...ProductPriceFragmentQuery\n            __typename\n        }\n        thumbnailUrl1x: thumbnail(size: 255){\n            url\n        }\n        thumbnailUrl2x:     thumbnail(size: 510){\n            url\n        }\n        __typename\n    }\n\n    fragment ProductPriceFragmentQuery on ProductPricingInfo {\n        discount {\n            gross {\n                amount\n                currency\n                __typename\n            }\n            __typename\n        }\n        priceRange {\n            stop {\n                gross {\n                    amount\n                    currency\n                    __typename\n                }\n                currency\n                __typename\n            }\n            start {\n                gross {\n                    amount\n                    currency\n                    __typename\n                }\n                currency\n                __typename\n            }\n            __typename\n        }\n        __typename\n    }\n\n    fragment ProductFiltersFragmentQuery on Attribute {\n        id\n        name\n        slug\n        choices(first: 10) {\n            edges {\n                node {\n                    id\n                    name\n                    slug\n                    __typename\n                }\n            }\n        }\n        __typename\n    }\n    '
    variables = {'categoryId': graphene.Node.to_global_id('Category', category.id), 'sortBy': {'field': 'NAME', 'direction': 'ASC'}, 'first': 1, 'attributesFilter': [{'slug': f'{product_attr.slug}', 'values': [f'{attr_value.slug}']}], 'channel': channel_USD.slug}
    response = user_api_client.post_graphql(query, variables)
    get_graphql_content(response)

def test_get_nodes(product_list):
    if False:
        print('Hello World!')
    global_ids = [to_global_id('Product', product.pk) for product in product_list]
    global_ids.append(to_global_id('Product', product_list[0].pk))
    products = get_nodes(global_ids, Product)
    assert products == product_list
    nonexistent_item = Mock(type='Product', pk=-1)
    nonexistent_item_global_id = to_global_id(nonexistent_item.type, nonexistent_item.pk)
    global_ids.append(nonexistent_item_global_id)
    msg = 'There is no node of type {} with pk {}'.format(nonexistent_item.type, nonexistent_item.pk)
    with pytest.raises(AssertionError) as exc:
        get_nodes(global_ids, Product)
    assert exc.value.args == (msg,)
    global_ids.pop()
    invalid_item = Mock(type='test', pk=-1)
    invalid_item_global_id = to_global_id(invalid_item.type, invalid_item.pk)
    global_ids.append(invalid_item_global_id)
    with pytest.raises(GraphQLError, match='Must receive Product id') as exc:
        get_nodes(global_ids, Product)
    assert exc.value.args == (f'Must receive Product id: {invalid_item_global_id}.',)
    global_ids = []
    with pytest.raises(GraphQLError, match='Could not resolve to a node with the global id list of'):
        get_nodes(global_ids, Product)
    global_ids = ['a', 'bb']
    with pytest.raises(GraphQLError, match='Could not resolve to a node with the global id list of'):
        get_nodes(global_ids, Product)

def test_get_nodes_for_order_with_int_id(order_list):
    if False:
        i = 10
        return i + 15
    order_models.Order.objects.update(use_old_id=True)
    global_ids = [to_global_id('Order', order.number) for order in order_list]
    global_ids.append(to_global_id('Order', order_list[0].number))
    orders = get_nodes(global_ids, Order)
    assert orders == order_list

def test_get_nodes_for_order_with_uuid_id(order_list):
    if False:
        print('Hello World!')
    global_ids = [to_global_id('Order', order.pk) for order in order_list]
    global_ids.append(to_global_id('Order', order_list[0].pk))
    orders = get_nodes(global_ids, Order)
    assert orders == order_list

def test_get_nodes_for_order_with_int_id_and_use_old_id_set_to_false(order_list):
    if False:
        print('Hello World!')
    'Test that `get_node` respects `use_old_id`.'
    global_ids = [to_global_id('Order', order.number) for order in order_list]
    global_ids.append(to_global_id('Order', order_list[0].pk))
    with pytest.raises(AssertionError):
        get_nodes(global_ids, Order)

def test_get_nodes_for_order_with_uuid_and_int_id(order_list):
    if False:
        return 10
    'Test that `get_nodes` works for both old and new order IDs.'
    order_models.Order.objects.update(use_old_id=True)
    global_ids = [to_global_id('Order', order.pk) for order in order_list[:-1]]
    global_ids.append(to_global_id('Order', order_list[-1].number))
    orders = get_nodes(global_ids, Order)
    assert orders == order_list

def test_from_global_id_or_error(product):
    if False:
        print('Hello World!')
    invalid_id = 'invalid'
    message = f'Invalid ID: {invalid_id}.'
    with pytest.raises(GraphQLError) as error:
        from_global_id_or_error(invalid_id)
    assert str(error.value) == message

def test_from_global_id_or_error_wth_invalid_type(product):
    if False:
        print('Hello World!')
    product_id = graphene.Node.to_global_id('Product', product.id)
    message = f'Invalid ID: {product_id}. Expected: ProductVariant, received: Product.'
    with pytest.raises(GraphQLError) as error:
        from_global_id_or_error(product_id, 'ProductVariant', raise_error=True)
    assert str(error.value) == message

def test_from_global_id_or_error_wth_type(product):
    if False:
        for i in range(10):
            print('nop')
    expected_product_type = str(Product)
    expected_product_id = graphene.Node.to_global_id(expected_product_type, product.id)
    (product_type, product_id) = from_global_id_or_error(expected_product_id, expected_product_type)
    assert product_id == str(product.id)
    assert product_type == expected_product_type

@mock.patch('saleor.graphql.order.schema.create_connection_slice')
def test_query_allow_replica(mocked_resolver, staff_api_client, order, permission_manage_orders):
    if False:
        return 10
    query = '\n        query {\n          orders(first: 5){\n            edges {\n              node {\n                id\n              }\n            }\n          }\n        }\n    '
    staff_api_client.post_graphql(query, permissions=[permission_manage_orders])
    assert mocked_resolver.call_args[0][1].context.allow_replica