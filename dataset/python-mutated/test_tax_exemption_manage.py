import graphene
from django.utils import timezone
from freezegun import freeze_time
from .....order import OrderStatus
from .....tax.error_codes import TaxExemptionManageErrorCode
from ....tests.utils import get_graphql_content
TAX_EXEMPTION_MUTATION = '\n    mutation manageTaxExemption($id: ID!, $taxExemption: Boolean!) {\n        taxExemptionManage(id: $id, taxExemption: $taxExemption) {\n            taxableObject {\n                ...on Checkout {\n                    id\n                    taxExemption\n                },\n                ...on Order {\n                    id\n                    taxExemption\n                }\n            }\n            errors {\n                field\n                code\n                message\n            }\n        }\n    }\n'

@freeze_time('2022-05-12 12:00:00')
def test_tax_exemption_manage_for_checkout_as_staff(staff_api_client, checkout, permission_manage_taxes):
    if False:
        i = 10
        return i + 15
    global_id = graphene.Node.to_global_id('Checkout', checkout.token)
    variables = {'id': global_id, 'taxExemption': True}
    response = staff_api_client.post_graphql(TAX_EXEMPTION_MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxExemptionManage']
    checkout.refresh_from_db()
    assert data['taxableObject']['id'] == global_id
    assert data['taxableObject']['taxExemption']
    assert checkout.price_expiration == timezone.now()

@freeze_time('2022-05-12 12:00:00')
def test_tax_exemption_manage_for_checkout_as_app(app_api_client, checkout, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    global_id = graphene.Node.to_global_id('Checkout', checkout.token)
    variables = {'id': global_id, 'taxExemption': True}
    response = app_api_client.post_graphql(TAX_EXEMPTION_MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxExemptionManage']
    checkout.refresh_from_db()
    assert data['taxableObject']['id'] == global_id
    assert data['taxableObject']['taxExemption']
    assert checkout.price_expiration == timezone.now()

def test_tax_exemption_manage_for_order(staff_api_client, order_unconfirmed, permission_manage_taxes):
    if False:
        return 10
    assert not order_unconfirmed.should_refresh_prices
    global_id = graphene.Node.to_global_id('Order', order_unconfirmed.id)
    variables = {'id': global_id, 'taxExemption': True}
    response = staff_api_client.post_graphql(TAX_EXEMPTION_MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxExemptionManage']
    order_unconfirmed.refresh_from_db()
    assert data['taxableObject']['id'] == global_id
    assert data['taxableObject']['taxExemption']
    assert order_unconfirmed.should_refresh_prices

def test_tax_exemption_manage_return_error_when_invalid_object_id(staff_api_client, product, permission_manage_taxes):
    if False:
        for i in range(10):
            print('nop')
    global_id = graphene.Node.to_global_id('Product', product.id)
    variables = {'id': global_id, 'taxExemption': True}
    response = staff_api_client.post_graphql(TAX_EXEMPTION_MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxExemptionManage']
    assert data['errors'][0]['field'] == 'id'
    assert data['errors'][0]['code'] == TaxExemptionManageErrorCode.NOT_FOUND.name
    assert not data['taxableObject']

def test_tax_exemption_manage_return_error_when_invalid_order_status(staff_api_client, order, permission_manage_taxes):
    if False:
        return 10
    order.status = OrderStatus.FULFILLED
    order.save(update_fields=['status'])
    global_id = graphene.Node.to_global_id('Order', order.id)
    variables = {'id': global_id, 'taxExemption': True}
    response = staff_api_client.post_graphql(TAX_EXEMPTION_MUTATION, variables, permissions=[permission_manage_taxes])
    content = get_graphql_content(response)
    data = content['data']['taxExemptionManage']
    assert data['errors']
    assert not data['taxableObject']
    assert data['errors'][0]['code'] == TaxExemptionManageErrorCode.NOT_EDITABLE_ORDER.name