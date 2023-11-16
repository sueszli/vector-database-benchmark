from ...utils import get_graphql_content
SALE_CREATE_CHANNEL_LISTING_MUTATION = '\nmutation SaleUpdate($id: ID!, $input: SaleChannelListingInput!) {\n  saleChannelListingUpdate(id: $id, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    sale {\n      id\n      name\n      channelListings {\n        channel {\n          id\n        }\n        discountValue\n      }\n    }\n  }\n}\n'

def raw_create_sale_channel_listing(staff_api_client, sale_id, add_channels=None, remove_channels=None):
    if False:
        i = 10
        return i + 15
    if not add_channels:
        add_channels = []
    if not remove_channels:
        remove_channels = []
    variables = {'id': sale_id, 'input': {'addChannels': add_channels, 'removeChannels': remove_channels}}
    response = staff_api_client.post_graphql(SALE_CREATE_CHANNEL_LISTING_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    return content

def create_sale_channel_listing(staff_api_client, sale_id, add_channels=None, remove_channels=None):
    if False:
        i = 10
        return i + 15
    response = raw_create_sale_channel_listing(staff_api_client, sale_id, add_channels=add_channels, remove_channels=remove_channels)
    assert response['data']['saleChannelListingUpdate']['errors'] == []
    data = response['data']['saleChannelListingUpdate']['sale']
    return data