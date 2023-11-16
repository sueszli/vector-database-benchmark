from google.cloud.video import stitcher_v1

def sample_create_cdn_key():
    if False:
        print('Hello World!')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.CreateCdnKeyRequest(parent='parent_value', cdn_key_id='cdn_key_id_value')
    operation = client.create_cdn_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)