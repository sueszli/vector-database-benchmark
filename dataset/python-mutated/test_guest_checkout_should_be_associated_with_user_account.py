import pytest
from ..product.utils.preparing_product import prepare_product
from ..shop.utils.preparing_shop import prepare_shop
from ..users.utils import create_customer, get_user
from ..utils import assign_permissions
from .utils import checkout_complete, checkout_create, checkout_delivery_method_update, checkout_dummy_payment_create

def create_active_customer(e2e_staff_api_client):
    if False:
        return 10
    email = 'user@saleor.io'
    user_data = create_customer(e2e_staff_api_client, email, is_active=True)
    user_id = user_data['id']
    return (user_id, email)

@pytest.mark.e2e
def test_guest_checkout_should_be_associated_with_user_account_CORE_1517(e2e_staff_api_client, e2e_not_logged_api_client, permission_manage_products, permission_manage_channels, permission_manage_shipping, permission_manage_product_types_and_attributes, permission_manage_orders, permission_manage_checkouts, permission_manage_users, permission_manage_settings):
    if False:
        print('Hello World!')
    permissions = [permission_manage_products, permission_manage_channels, permission_manage_shipping, permission_manage_product_types_and_attributes, permission_manage_orders, permission_manage_checkouts, permission_manage_users, permission_manage_settings]
    assign_permissions(e2e_staff_api_client, permissions)
    (warehouse_id, channel_id, channel_slug, shipping_method_id) = prepare_shop(e2e_staff_api_client)
    variant_price = 10
    (_product_id, product_variant_id, _product_variant_price) = prepare_product(e2e_staff_api_client, warehouse_id, channel_id, variant_price)
    (user_id, email) = create_active_customer(e2e_staff_api_client)
    lines = [{'variantId': product_variant_id, 'quantity': 1}]
    checkout_data = checkout_create(e2e_not_logged_api_client, lines, channel_slug, email, set_default_billing_address=True, set_default_shipping_address=True)
    checkout_id = checkout_data['id']
    assert checkout_data['isShippingRequired'] is True
    shipping_method_id = checkout_data['shippingMethods'][0]['id']
    assert checkout_data['deliveryMethod'] is None
    checkout_data = checkout_delivery_method_update(e2e_not_logged_api_client, checkout_id, shipping_method_id)
    assert checkout_data['deliveryMethod']['id'] == shipping_method_id
    total_gross_amount = checkout_data['totalPrice']['gross']['amount']
    checkout_dummy_payment_create(e2e_not_logged_api_client, checkout_id, total_gross_amount)
    order_data = checkout_complete(e2e_not_logged_api_client, checkout_id)
    order_id = order_data['id']
    assert order_data['isShippingRequired'] is True
    assert order_data['status'] == 'UNFULFILLED'
    assert order_data['total']['gross']['amount'] == total_gross_amount
    assert order_data['deliveryMethod']['id'] == shipping_method_id
    data = get_user(e2e_staff_api_client, user_id)
    assert data['id'] == user_id
    assert data['orders']['edges'][0]['node']['id'] == order_id