from ...utils import get_graphql_content
COLLECTION_CHANNEL_LISTING_UPDATE_MUTATION = '\nmutation CollectionListing($id: ID!, $input: CollectionChannelListingUpdateInput!) {\n  collectionChannelListingUpdate(id: $id, input: $input) {\n    errors {\n      code\n      field\n      message\n      channels\n    }\n    collection {\n      id\n      channelListings {\n        id\n        isPublished\n        publishedAt\n        channel {\n          id\n        }\n      }\n    }\n  }\n}\n'

def create_collection_channel_listing(staff_api_client, collection_id, channel_id, publication_date=None, is_published=False):
    if False:
        return 10
    variables = {'id': collection_id, 'input': {'addChannels': [{'channelId': channel_id, 'isPublished': is_published, 'publicationDate': publication_date}], 'removeChannels': []}}
    response = staff_api_client.post_graphql(COLLECTION_CHANNEL_LISTING_UPDATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['collectionChannelListingUpdate']['collection']
    assert content['data']['collectionChannelListingUpdate']['errors'] == []
    assert data['channelListings'][0]['channel']['id'] == channel_id
    return data