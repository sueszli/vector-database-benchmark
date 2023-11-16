from google.cloud import channel_v1

def sample_list_transferable_skus():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListTransferableSkusRequest(cloud_identity_id='cloud_identity_id_value', parent='parent_value')
    page_result = client.list_transferable_skus(request=request)
    for response in page_result:
        print(response)