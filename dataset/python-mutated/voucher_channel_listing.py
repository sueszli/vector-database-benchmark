from ...utils import get_graphql_content
VOUCHER_CREATE_CHANNEL_LISTING_MUTATION = '\nmutation VoucherChannelListingUpdate($id: ID!, $input: VoucherChannelListingInput!) {\n  voucherChannelListingUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    voucher {\n      id\n      startDate\n      discountValueType\n      type\n      channelListings {\n        id\n        channel {\n          id\n        }\n        discountValue\n        currency\n        minSpent {\n          amount\n        }\n      }\n    }\n  }\n}\n'

def create_voucher_channel_listing(staff_api_client, voucher_id, addChannels=None, removeChannels=None):
    if False:
        while True:
            i = 10
    if not addChannels:
        addChannels = []
    if not removeChannels:
        removeChannels = []
    variables = {'id': voucher_id, 'input': {'addChannels': addChannels, 'removeChannels': removeChannels}}
    response = staff_api_client.post_graphql(VOUCHER_CREATE_CHANNEL_LISTING_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    assert content['data']['voucherChannelListingUpdate']['errors'] == []
    data = content['data']['voucherChannelListingUpdate']['voucher']
    return data