import graphene
from .....order import OrderStatus
from .....order import models as order_models
from ....tests.utils import assert_no_permission, get_graphql_content
DRAFT_ORDER_BULK_DELETE = '\n    mutation draftOrderBulkDelete($ids: [ID!]!) {\n        draftOrderBulkDelete(ids: $ids) {\n            count\n        }\n    }\n'

def test_delete_draft_orders(staff_api_client, order_list, permission_group_manage_orders):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    (order_1, order_2, *orders) = order_list
    order_1.status = OrderStatus.DRAFT
    order_2.status = OrderStatus.DRAFT
    order_1.save()
    order_2.save()
    query = DRAFT_ORDER_BULK_DELETE
    variables = {'ids': [graphene.Node.to_global_id('Order', order.id) for order in order_list]}
    response = staff_api_client.post_graphql(query, variables)
    content = get_graphql_content(response)
    assert content['data']['draftOrderBulkDelete']['count'] == 2
    assert not order_models.Order.objects.filter(id__in=[order_1.id, order_2.id]).exists()
    assert order_models.Order.objects.filter(id__in=[order.id for order in orders]).count() == len(orders)

def test_delete_draft_orders_by_user_no_channel_access(staff_api_client, order_list, permission_group_all_perms_channel_USD_only, channel_PLN):
    if False:
        i = 10
        return i + 15
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    (order_1, order_2, *orders) = order_list
    order_1.status = OrderStatus.DRAFT
    order_1.channel = channel_PLN
    order_2.status = OrderStatus.DRAFT
    order_1.save()
    order_2.save()
    query = DRAFT_ORDER_BULK_DELETE
    variables = {'ids': [graphene.Node.to_global_id('Order', order.id) for order in order_list]}
    response = staff_api_client.post_graphql(query, variables)
    assert_no_permission(response)

def test_delete_draft_orders_by_app(app_api_client, order_list, permission_manage_orders):
    if False:
        while True:
            i = 10
    (order_1, order_2, *orders) = order_list
    order_1.status = OrderStatus.DRAFT
    order_2.status = OrderStatus.DRAFT
    order_1.save()
    order_2.save()
    query = DRAFT_ORDER_BULK_DELETE
    variables = {'ids': [graphene.Node.to_global_id('Order', order.id) for order in order_list]}
    response = app_api_client.post_graphql(query, variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    assert content['data']['draftOrderBulkDelete']['count'] == 2
    assert not order_models.Order.objects.filter(id__in=[order_1.id, order_2.id]).exists()
    assert order_models.Order.objects.filter(id__in=[order.id for order in orders]).count() == len(orders)
MUTATION_DELETE_ORDER_LINES = '\n    mutation draftOrderLinesBulkDelete($ids: [ID!]!) {\n        draftOrderLinesBulkDelete(ids: $ids) {\n            count\n            errors {\n                field\n                message\n            }\n        }\n    }\n'

def test_fail_to_delete_non_draft_order_lines(staff_api_client, order_with_lines, permission_group_manage_orders):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = order_with_lines
    order_lines = [line for line in order.lines.all()]
    order.status = OrderStatus.CANCELED
    order.save()
    variables = {'ids': [graphene.Node.to_global_id('OrderLine', order_line.id) for order_line in order_lines]}
    response = staff_api_client.post_graphql(MUTATION_DELETE_ORDER_LINES, variables)
    content = get_graphql_content(response)
    assert 'errors' in content['data']['draftOrderLinesBulkDelete']
    assert content['data']['draftOrderLinesBulkDelete']['count'] == 0

def test_delete_draft_order_lines(staff_api_client, order_with_lines, permission_group_manage_orders):
    if False:
        print('Hello World!')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    order = order_with_lines
    order_lines = [line for line in order.lines.all()]
    order.status = OrderStatus.DRAFT
    order.save()
    variables = {'ids': [graphene.Node.to_global_id('OrderLine', order_line.id) for order_line in order_lines]}
    response = staff_api_client.post_graphql(MUTATION_DELETE_ORDER_LINES, variables)
    content = get_graphql_content(response)
    assert content['data']['draftOrderLinesBulkDelete']['count'] == 2
    assert not order_models.OrderLine.objects.filter(id__in=[order_line.pk for order_line in order_lines]).exists()

def test_delete_draft_order_lines_by_user_no_channel_access(staff_api_client, order_with_lines, permission_group_all_perms_channel_USD_only, channel_PLN):
    if False:
        return 10
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    order = order_with_lines
    order_lines = [line for line in order.lines.all()]
    order.status = OrderStatus.DRAFT
    order.channel = channel_PLN
    order.save(update_fields=['status', 'channel'])
    variables = {'ids': [graphene.Node.to_global_id('OrderLine', order_line.id) for order_line in order_lines]}
    response = staff_api_client.post_graphql(MUTATION_DELETE_ORDER_LINES, variables)
    assert_no_permission(response)

def test_delete_draft_order_lines_by_app(app_api_client, order_with_lines, permission_manage_orders):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    order_lines = [line for line in order.lines.all()]
    order.status = OrderStatus.DRAFT
    order.save()
    variables = {'ids': [graphene.Node.to_global_id('OrderLine', order_line.id) for order_line in order_lines]}
    response = app_api_client.post_graphql(MUTATION_DELETE_ORDER_LINES, variables, permissions=(permission_manage_orders,))
    content = get_graphql_content(response)
    assert content['data']['draftOrderLinesBulkDelete']['count'] == 2
    assert not order_models.OrderLine.objects.filter(id__in=[order_line.pk for order_line in order_lines]).exists()