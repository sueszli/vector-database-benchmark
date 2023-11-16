from google.cloud.video import live_stream_v1

def sample_create_asset():
    if False:
        while True:
            i = 10
    client = live_stream_v1.LivestreamServiceClient()
    request = live_stream_v1.CreateAssetRequest(parent='parent_value', asset_id='asset_id_value')
    operation = client.create_asset(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)