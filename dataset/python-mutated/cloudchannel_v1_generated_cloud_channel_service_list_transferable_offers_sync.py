from google.cloud import channel_v1

def sample_list_transferable_offers():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListTransferableOffersRequest(cloud_identity_id='cloud_identity_id_value', parent='parent_value', sku='sku_value')
    page_result = client.list_transferable_offers(request=request)
    for response in page_result:
        print(response)