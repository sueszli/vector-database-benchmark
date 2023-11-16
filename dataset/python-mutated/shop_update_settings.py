from ...utils import get_graphql_content
SHOP_SETTING_UPDATE_MUTATION = '\nmutation ShopSettingsUpdate($input: ShopSettingsInput!) {\n  shopSettingsUpdate(input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    shop {\n      enableAccountConfirmationByEmail\n      fulfillmentAutoApprove\n      fulfillmentAllowUnpaid\n    }\n  }\n}\n'

def update_shop_settings(staff_api_client, input):
    if False:
        for i in range(10):
            print('nop')
    variables = {'input': input}
    response = staff_api_client.post_graphql(SHOP_SETTING_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['shopSettingsUpdate']['errors'] == []
    data = content['data']['shopSettingsUpdate']['shop']
    return data