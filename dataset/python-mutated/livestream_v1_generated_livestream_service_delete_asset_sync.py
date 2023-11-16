from google.cloud.video import live_stream_v1

def sample_delete_asset():
    if False:
        i = 10
        return i + 15
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.DeleteAssetRequest(name='name_value')
    operation = client.delete_asset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)