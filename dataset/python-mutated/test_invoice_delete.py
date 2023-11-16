from unittest.mock import patch
import graphene
from ....graphql.tests.utils import assert_no_permission, get_graphql_content
from ....invoice.error_codes import InvoiceErrorCode
from ....invoice.models import Invoice, InvoiceEvent, InvoiceEvents
INVOICE_DELETE_MUTATION = '\n    mutation invoiceDelete($id: ID!) {\n        invoiceDelete(\n            id: $id\n        ) {\n            errors {\n                field\n                code\n            }\n        }\n    }\n'

def test_invoice_delete(staff_api_client, permission_group_manage_orders, order):
    if False:
        for i in range(10):
            print('nop')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    response = staff_api_client.post_graphql(INVOICE_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    assert not content['data']['invoiceDelete']['errors']
    assert not Invoice.objects.filter(id=invoice.pk).exists()
    assert InvoiceEvent.objects.filter(type=InvoiceEvents.DELETED, user=staff_api_client.user, parameters__invoice_id=invoice.id).exists()

def test_invoice_delete_by_user_no_channel_access(staff_api_client, permission_group_all_perms_channel_USD_only, order, channel_PLN):
    if False:
        while True:
            i = 10
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    order.channel = channel_PLN
    order.save(update_fields=['channel'])
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    response = staff_api_client.post_graphql(INVOICE_DELETE_MUTATION, variables)
    assert_no_permission(response)

def test_invoice_delete_by_app(app_api_client, permission_manage_orders, order):
    if False:
        i = 10
        return i + 15
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    response = app_api_client.post_graphql(INVOICE_DELETE_MUTATION, variables, permissions=[permission_manage_orders])
    content = get_graphql_content(response)
    assert not content['data']['invoiceDelete']['errors']
    assert not Invoice.objects.filter(id=invoice.pk).exists()
    assert InvoiceEvent.objects.filter(type=InvoiceEvents.DELETED, user=None, app=app_api_client.app, parameters__invoice_id=invoice.id).exists()

@patch('saleor.plugins.manager.PluginsManager.invoice_delete')
def test_invoice_delete_invalid_id(plugin_mock, staff_api_client, permission_group_manage_orders):
    if False:
        for i in range(10):
            print('nop')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    variables = {'id': graphene.Node.to_global_id('Invoice', 1337)}
    response = staff_api_client.post_graphql(INVOICE_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    error = content['data']['invoiceDelete']['errors'][0]
    assert error['code'] == InvoiceErrorCode.NOT_FOUND.name
    assert error['field'] == 'id'
    plugin_mock.assert_not_called()