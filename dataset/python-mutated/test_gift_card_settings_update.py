from .....core import TimePeriodType
from .....site import GiftCardSettingsExpiryType
from .....site.error_codes import GiftCardSettingsErrorCode
from ....core.enums import TimePeriodTypeEnum
from ....tests.utils import assert_no_permission, get_graphql_content
from ...enums import GiftCardSettingsExpiryTypeEnum
GIFT_CARD_SETTINGS_UPDATE_MUTATION = '\n    mutation giftCardSettingsUpdate($input: GiftCardSettingsUpdateInput!) {\n        giftCardSettingsUpdate(input: $ input) {\n            giftCardSettings {\n                expiryType\n                expiryPeriod {\n                    type\n                    amount\n                }\n            }\n            errors {\n                code\n                field\n            }\n        }\n    }\n'

def test_gift_card_settings_update_by_staff(staff_api_client, site_settings, permission_manage_gift_card):
    if False:
        for i in range(10):
            print('nop')
    assert site_settings.gift_card_expiry_type == GiftCardSettingsExpiryType.NEVER_EXPIRE
    expiry_type = GiftCardSettingsExpiryTypeEnum.EXPIRY_PERIOD.name
    expiry_period_type = TimePeriodTypeEnum.DAY.name
    expiry_period = 50
    variables = {'input': {'expiryType': expiry_type, 'expiryPeriod': {'type': expiry_period_type, 'amount': expiry_period}}}
    response = staff_api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardSettingsUpdate']
    assert not data['errors']
    assert data['giftCardSettings']
    assert data['giftCardSettings']['expiryType'] == expiry_type
    assert data['giftCardSettings']['expiryPeriod']['type'] == expiry_period_type
    assert data['giftCardSettings']['expiryPeriod']['amount'] == expiry_period

def test_gift_card_settings_update_by_app(app_api_client, site_settings, permission_manage_gift_card):
    if False:
        for i in range(10):
            print('nop')
    assert site_settings.gift_card_expiry_type == GiftCardSettingsExpiryType.NEVER_EXPIRE
    expiry_type = GiftCardSettingsExpiryTypeEnum.EXPIRY_PERIOD.name
    expiry_period_type = TimePeriodTypeEnum.DAY.name
    expiry_period = 50
    variables = {'input': {'expiryType': expiry_type, 'expiryPeriod': {'type': expiry_period_type, 'amount': expiry_period}}}
    response = app_api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardSettingsUpdate']
    assert not data['errors']
    assert data['giftCardSettings']
    assert data['giftCardSettings']['expiryType'] == expiry_type
    assert data['giftCardSettings']['expiryPeriod']['type'] == expiry_period_type
    assert data['giftCardSettings']['expiryPeriod']['amount'] == expiry_period

def test_gift_card_settings_update_by_customer(api_client, site_settings):
    if False:
        for i in range(10):
            print('nop')
    assert site_settings.gift_card_expiry_type == GiftCardSettingsExpiryType.NEVER_EXPIRE
    expiry_type = GiftCardSettingsExpiryTypeEnum.EXPIRY_PERIOD.name
    expiry_period_type = TimePeriodTypeEnum.DAY.name
    expiry_period = 50
    variables = {'input': {'expiryType': expiry_type, 'expiryPeriod': {'type': expiry_period_type, 'amount': expiry_period}}}
    response = api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables)
    assert_no_permission(response)

def test_gift_card_settings_update_with_the_same_type(staff_api_client, site_settings, permission_manage_gift_card):
    if False:
        for i in range(10):
            print('nop')
    assert site_settings.gift_card_expiry_type == GiftCardSettingsExpiryType.NEVER_EXPIRE
    expiry_type = GiftCardSettingsExpiryTypeEnum.NEVER_EXPIRE.name
    variables = {'input': {'expiryType': expiry_type}}
    response = staff_api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardSettingsUpdate']
    assert not data['errors']
    assert data['giftCardSettings']
    assert data['giftCardSettings']['expiryType'] == expiry_type
    assert data['giftCardSettings']['expiryPeriod'] is None

def test_gift_card_settings_update_change_to_expiry_period_no_data_given(staff_api_client, site_settings, permission_manage_gift_card):
    if False:
        while True:
            i = 10
    assert site_settings.gift_card_expiry_type == GiftCardSettingsExpiryType.NEVER_EXPIRE
    expiry_type = GiftCardSettingsExpiryTypeEnum.EXPIRY_PERIOD.name
    variables = {'input': {'expiryType': expiry_type}}
    response = staff_api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardSettingsUpdate']
    assert not data['giftCardSettings']
    assert len(data['errors']) == 1
    assert data['errors'][0]['field'] == 'expiryPeriod'
    assert data['errors'][0]['code'] == GiftCardSettingsErrorCode.REQUIRED.name

def test_gift_card_settings_update_change_to_never_expire(staff_api_client, site_settings, permission_manage_gift_card):
    if False:
        i = 10
        return i + 15
    site_settings.gift_card_expiry_type = GiftCardSettingsExpiryType.EXPIRY_PERIOD
    site_settings.gift_card_expiry_period_type = TimePeriodType.MONTH
    site_settings.gift_card_expiry_period = 10
    expiry_type = GiftCardSettingsExpiryTypeEnum.NEVER_EXPIRE.name
    variables = {'input': {'expiryType': expiry_type}}
    response = staff_api_client.post_graphql(GIFT_CARD_SETTINGS_UPDATE_MUTATION, variables, permissions=(permission_manage_gift_card,))
    content = get_graphql_content(response)
    data = content['data']['giftCardSettingsUpdate']
    assert not data['errors']
    assert data['giftCardSettings']
    assert data['giftCardSettings']['expiryType'] == expiry_type
    assert data['giftCardSettings']['expiryPeriod'] is None