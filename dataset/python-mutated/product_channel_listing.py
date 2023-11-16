import datetime
import pytz
from ...utils import get_graphql_content
PRODUCT_CHANNEL_LISTING_UPDATE_MUTATION = '\nmutation UpdateProductChannelListing(\n    $productId: ID!, $input: ProductChannelListingUpdateInput!\n) {\n  productChannelListingUpdate(id: $productId, input: $input) {\n    errors {\n      field\n      message\n      code\n    }\n    product {\n      id\n      channelListings {\n        id\n        isPublished\n        publicationDate\n        visibleInListings\n        isAvailableForPurchase\n        availableForPurchaseAt\n        channel {\n          id\n        }\n      }\n    }\n  }\n}\n'

def raw_create_product_channel_listing(staff_api_client, product_id, channel_id, publication_date=None, is_published=False, visible_in_listings=False, available_for_purchase_datetime=None, is_available_for_purchase=None):
    if False:
        print('Hello World!')
    variables = {'productId': product_id, 'input': {'updateChannels': [{'channelId': channel_id, 'isPublished': is_published, 'publicationDate': publication_date, 'visibleInListings': visible_in_listings, 'isAvailableForPurchase': is_available_for_purchase, 'availableForPurchaseAt': available_for_purchase_datetime}]}}
    response = staff_api_client.post_graphql(PRODUCT_CHANNEL_LISTING_UPDATE_MUTATION, variables, check_no_permissions=False)
    content = get_graphql_content(response)
    data = content['data']['productChannelListingUpdate']
    return data

def create_product_channel_listing(staff_api_client, product_id, channel_id, publication_date=datetime.date(2007, 1, 1), is_published=True, visible_in_listings=True, available_for_purchase_datetime=datetime.datetime(2007, 1, 1, tzinfo=pytz.utc), is_available_for_purchase=True):
    if False:
        i = 10
        return i + 15
    response = raw_create_product_channel_listing(staff_api_client, product_id, channel_id, publication_date, is_published, visible_in_listings, available_for_purchase_datetime, is_available_for_purchase)
    data = response['product']
    assert response['errors'] == []
    assert data['id'] == product_id
    channel_listing_data = data['channelListings'][0]
    assert channel_listing_data['channel']['id'] == channel_id
    assert channel_listing_data['isPublished'] is is_published
    assert channel_listing_data['publicationDate'] == publication_date.isoformat()
    assert channel_listing_data['visibleInListings'] is visible_in_listings
    assert channel_listing_data['isAvailableForPurchase'] is is_available_for_purchase
    assert channel_listing_data['availableForPurchaseAt'] == available_for_purchase_datetime.isoformat()
    return data