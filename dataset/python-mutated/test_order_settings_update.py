from ....tests.utils import assert_no_permission, get_graphql_content
ORDER_SETTINGS_UPDATE_MUTATION = '\n    mutation orderSettings($confirmOrders: Boolean, $fulfillGiftCards: Boolean) {\n        orderSettingsUpdate(\n            input: {\n                automaticallyConfirmAllNewOrders: $confirmOrders\n                automaticallyFulfillNonShippableGiftCard: $fulfillGiftCards\n            }\n        ) {\n            orderSettings {\n                automaticallyConfirmAllNewOrders\n                automaticallyFulfillNonShippableGiftCard\n                markAsPaidStrategy\n            }\n        }\n    }\n'

def test_order_settings_update_by_staff(staff_api_client, permission_group_manage_orders, channel_USD, channel_PLN):
    if False:
        for i in range(10):
            print('nop')
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    response = staff_api_client.post_graphql(ORDER_SETTINGS_UPDATE_MUTATION, {'confirmOrders': False, 'fulfillGiftCards': False})
    content = get_graphql_content(response)
    response_settings = content['data']['orderSettingsUpdate']['orderSettings']
    assert response_settings['automaticallyConfirmAllNewOrders'] is False
    assert response_settings['automaticallyFulfillNonShippableGiftCard'] is False
    channel_PLN.refresh_from_db()
    channel_USD.refresh_from_db()
    assert channel_PLN.automatically_confirm_all_new_orders is False
    assert channel_PLN.automatically_fulfill_non_shippable_gift_card is False
    assert channel_USD.automatically_confirm_all_new_orders is False
    assert channel_USD.automatically_fulfill_non_shippable_gift_card is False

def test_order_settings_update_by_staff_no_channel_access(staff_api_client, permission_group_all_perms_channel_USD_only, channel_USD, channel_PLN):
    if False:
        while True:
            i = 10
    permission_group_all_perms_channel_USD_only.user_set.add(staff_api_client.user)
    channel_USD.is_active = False
    channel_USD.save(update_fields=['is_active'])
    response = staff_api_client.post_graphql(ORDER_SETTINGS_UPDATE_MUTATION, {'confirmOrders': False, 'fulfillGiftCards': False})
    assert_no_permission(response)

def test_order_settings_update_by_staff_nothing_changed(staff_api_client, permission_group_manage_orders, channel_USD, channel_PLN):
    if False:
        i = 10
        return i + 15
    permission_group_manage_orders.user_set.add(staff_api_client.user)
    QUERY = '\n        mutation {\n            orderSettingsUpdate(\n                input: {}\n            ) {\n                orderSettings {\n                    automaticallyConfirmAllNewOrders\n                    automaticallyFulfillNonShippableGiftCard\n                }\n            }\n        }\n    '
    response = staff_api_client.post_graphql(QUERY, {})
    content = get_graphql_content(response)
    response_settings = content['data']['orderSettingsUpdate']['orderSettings']
    assert response_settings['automaticallyConfirmAllNewOrders'] is True
    assert response_settings['automaticallyFulfillNonShippableGiftCard'] is True
    channel_PLN.refresh_from_db()
    channel_USD.refresh_from_db()
    assert channel_PLN.automatically_confirm_all_new_orders is True
    assert channel_PLN.automatically_fulfill_non_shippable_gift_card is True
    assert channel_USD.automatically_confirm_all_new_orders is True
    assert channel_USD.automatically_fulfill_non_shippable_gift_card is True

def test_order_settings_update_by_app(app_api_client, permission_manage_orders, channel_USD, channel_PLN):
    if False:
        while True:
            i = 10
    app_api_client.app.permissions.set([permission_manage_orders])
    response = app_api_client.post_graphql(ORDER_SETTINGS_UPDATE_MUTATION, {'confirmOrders': False, 'fulfillGiftCards': False})
    content = get_graphql_content(response)
    response_settings = content['data']['orderSettingsUpdate']['orderSettings']
    assert response_settings['automaticallyConfirmAllNewOrders'] is False
    assert response_settings['automaticallyFulfillNonShippableGiftCard'] is False
    channel_PLN.refresh_from_db()
    channel_USD.refresh_from_db()
    assert channel_PLN.automatically_confirm_all_new_orders is False
    assert channel_PLN.automatically_fulfill_non_shippable_gift_card is False
    assert channel_USD.automatically_confirm_all_new_orders is False
    assert channel_USD.automatically_fulfill_non_shippable_gift_card is False

def test_order_settings_update_by_user_without_permissions(user_api_client, channel_USD, channel_PLN):
    if False:
        i = 10
        return i + 15
    response = user_api_client.post_graphql(ORDER_SETTINGS_UPDATE_MUTATION, {'confirmOrders': False, 'fulfillGiftCards': False})
    assert_no_permission(response)
    channel_PLN.refresh_from_db()
    channel_USD.refresh_from_db()
    assert channel_PLN.automatically_confirm_all_new_orders is True
    assert channel_PLN.automatically_fulfill_non_shippable_gift_card is True
    assert channel_USD.automatically_confirm_all_new_orders is True
    assert channel_USD.automatically_fulfill_non_shippable_gift_card is True