from ...utils import get_graphql_content
SHIPPING_METHOD_CHANNEL_LISTING_MUTATION = '\nmutation ShippingMethodChannelListingUpdate(\n    $id: ID!, $input: ShippingMethodChannelListingInput!\n) {\n  shippingMethodChannelListingUpdate(id: $id, input: $input) {\n    errors {\n      field\n      code\n      message\n    }\n    shippingMethod {\n        id\n        channelListings {\n            minimumOrderPrice {\n                amount\n             }\n        }\n    }\n  }\n}\n'

def create_shipping_method_channel_listing(staff_api_client, shipping_method_id, channel_id, price='10.00', minimumOrderPrice=None, maximumOrderPrice=None):
    if False:
        i = 10
        return i + 15
    variables = {'id': shipping_method_id, 'input': {'addChannels': [{'channelId': channel_id, 'price': price, 'maximumOrderPrice': maximumOrderPrice, 'minimumOrderPrice': minimumOrderPrice}]}}
    response = staff_api_client.post_graphql(SHIPPING_METHOD_CHANNEL_LISTING_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['shippingMethodChannelListingUpdate']['errors'] == []
    data = content['data']['shippingMethodChannelListingUpdate']['shippingMethod']
    assert data['id'] is not None
    return data