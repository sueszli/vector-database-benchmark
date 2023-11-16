from ....tests.utils import get_graphql_content
QUERY_GET_VARIANTS_FROM_ORDER = '\n{\n  me{\n    orders(first:10){\n      edges{\n        node{\n          lines{\n            variant{\n              id\n            }\n          }\n        }\n      }\n    }\n  }\n}\n'

def test_get_variant_from_order_line_variant_published_as_customer(user_api_client, order_line):
    if False:
        print('Hello World!')
    response = user_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {})
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_published_as_admin(staff_api_client, order_line, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    order = order_line.order
    order.user = staff_api_client.user
    order.save()
    response = staff_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {}, permissions=(permission_manage_products,), check_no_permissions=False)
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_not_published_as_customer(user_api_client, order_line):
    if False:
        for i in range(10):
            print('nop')
    product = order_line.variant.product
    product.channel_listings.update(is_published=False)
    response = user_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {})
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant'] is None

def test_get_variant_from_order_line_variant_not_published_as_admin(staff_api_client, order_line, permission_manage_products):
    if False:
        for i in range(10):
            print('nop')
    order = order_line.order
    order.user = staff_api_client.user
    order.save()
    product = order_line.variant.product
    product.channel_listings.update(is_published=False)
    response = staff_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {}, permissions=(permission_manage_products,), check_no_permissions=False)
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_not_assigned_to_channel_as_customer(user_api_client, order_line):
    if False:
        while True:
            i = 10
    product = order_line.variant.product
    product.channel_listings.all().delete()
    response = user_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {})
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant'] is None

def test_get_variant_from_order_line_variant_not_assigned_to_channel_as_admin(staff_api_client, order_line, permission_manage_products):
    if False:
        while True:
            i = 10
    order = order_line.order
    order.user = staff_api_client.user
    order.save()
    product = order_line.variant.product
    product.channel_listings.all().delete()
    response = staff_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {}, permissions=(permission_manage_products,), check_no_permissions=False)
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_not_visible_in_listings_as_customer(user_api_client, order_line):
    if False:
        return 10
    product = order_line.variant.product
    product.channel_listings.update(visible_in_listings=False)
    response = user_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {})
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_not_visible_in_listings_as_admin(staff_api_client, order_line, permission_manage_products):
    if False:
        i = 10
        return i + 15
    order = order_line.order
    order.user = staff_api_client.user
    order.save()
    product = order_line.variant.product
    product.channel_listings.update(visible_in_listings=False)
    response = staff_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {}, permissions=(permission_manage_products,), check_no_permissions=False)
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant']['id']

def test_get_variant_from_order_line_variant_not_exists_as_customer(user_api_client, order_line):
    if False:
        print('Hello World!')
    order_line.variant = None
    order_line.save()
    response = user_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {})
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant'] is None

def test_get_variant_from_order_line_variant_not_exists_as_staff(staff_api_client, order_line, permission_manage_products):
    if False:
        return 10
    order = order_line.order
    order.user = staff_api_client.user
    order.save()
    order_line.variant = None
    order_line.save()
    response = staff_api_client.post_graphql(QUERY_GET_VARIANTS_FROM_ORDER, {}, permissions=(permission_manage_products,), check_no_permissions=False)
    content = get_graphql_content(response)
    orders = content['data']['me']['orders']['edges']
    assert orders[0]['node']['lines'][0]['variant'] is None