from ...utils import get_graphql_content
CHANNEL_UPDATE_MUTATION = '\nmutation ChannelUpdate($id: ID!, $input: ChannelUpdateInput!) {\n  channelUpdate(id: $id, input: $input) {\n    channel {\n      id\n      orderSettings {\n        deleteExpiredOrdersAfter\n        allowUnpaidOrders\n        automaticallyFulfillNonShippableGiftCard\n        automaticallyConfirmAllNewOrders\n        expireOrdersAfter\n        deleteExpiredOrdersAfter\n      }\n    }\n    errors {\n      message\n      field\n    }\n  }\n}\n\n'

def update_channel(staff_api_client, id, input):
    if False:
        while True:
            i = 10
    variables = {'id': id, 'input': input}
    response = staff_api_client.post_graphql(CHANNEL_UPDATE_MUTATION, variables)
    content = get_graphql_content(response)
    assert content['data']['channelUpdate']['errors'] == []
    data = content['data']['channelUpdate']['channel']
    assert data['id'] is not None
    return data