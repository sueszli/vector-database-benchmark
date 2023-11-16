from unittest.mock import patch
import graphene
import pytest
from ....graphql.tests.utils import assert_no_permission, get_graphql_content
from ....invoice.error_codes import InvoiceErrorCode
from ....invoice.models import Invoice, InvoiceEvent, InvoiceEvents
INVOICE_REQUEST_DELETE_MUTATION = '\n    mutation invoiceRequestDelete($id: ID!) {\n        invoiceRequestDelete(\n            id: $id\n        ) {\n            errors {\n                field\n                code\n            }\n        }\n    }\n'

@pytest.fixture(autouse=True)
def setup_dummy_gateways(settings):
    if False:
        while True:
            i = 10
    settings.PLUGINS = ['saleor.payment.gateways.dummy.plugin.DummyGatewayPlugin']
    return settings

@patch('saleor.plugins.manager.PluginsManager.invoice_delete')
def test_invoice_request_delete(plugin_mock, staff_api_client, permission_group_manage_orders, order):
    if False:
        while True:
            i = 10
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    staff_api_client.post_graphql(INVOICE_REQUEST_DELETE_MUTATION, variables)
    invoice.refresh_from_db()
    plugin_mock.assert_called_once_with(invoice)
    assert InvoiceEvent.objects.filter(type=InvoiceEvents.REQUESTED_DELETION, user=staff_api_client.user, invoice=invoice, order=invoice.order).exists()

def test_invoice_request_delete_by_user_no_channel_access(staff_api_client, permission_group_all_perms_channel_USD_only, order, channel_PLN):
    if False:
        for i in range(10):
            print('nop')
    order.channel = channel_PLN
    order.save(update_fields=['channel'])
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(INVOICE_REQUEST_DELETE_MUTATION, variables)
    assert_no_permission(response)

@patch('saleor.plugins.manager.PluginsManager.invoice_delete')
def test_invoice_request_delete_by_app(plugin_mock, app_api_client, permission_manage_orders, order):
    if False:
        print('Hello World!')
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    app_api_client.post_graphql(INVOICE_REQUEST_DELETE_MUTATION, variables, permissions=(permission_manage_orders,))
    invoice.refresh_from_db()
    plugin_mock.assert_called_once_with(invoice)
    assert InvoiceEvent.objects.filter(type=InvoiceEvents.REQUESTED_DELETION, user=None, app=app_api_client.app, invoice=invoice, order=invoice.order).exists()

@patch('saleor.plugins.manager.PluginsManager.invoice_delete')
def test_invoice_request_delete_invalid_id(plugin_mock, staff_api_client, permission_group_manage_orders):
    if False:
        return 10
    variables = {'id': graphene.Node.to_global_id('Invoice', 1337)}
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(INVOICE_REQUEST_DELETE_MUTATION, variables)
    content = get_graphql_content(response)
    error = content['data']['invoiceRequestDelete']['errors'][0]
    assert error['code'] == InvoiceErrorCode.NOT_FOUND.name
    assert error['field'] == 'id'
    plugin_mock.assert_not_called()

@patch('saleor.plugins.manager.PluginsManager.invoice_delete')
def test_invoice_request_delete_no_permission(plugin_mock, staff_api_client, permission_group_manage_orders, order):
    if False:
        print('Hello World!')
    invoice = Invoice.objects.create(order=order)
    variables = {'id': graphene.Node.to_global_id('Invoice', invoice.pk)}
    response = staff_api_client.post_graphql(INVOICE_REQUEST_DELETE_MUTATION, variables)
    assert_no_permission(response)
    plugin_mock.assert_not_called()