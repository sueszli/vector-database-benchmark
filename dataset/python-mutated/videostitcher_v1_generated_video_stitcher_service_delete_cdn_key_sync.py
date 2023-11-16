from google.cloud.video import stitcher_v1

def sample_delete_cdn_key():
    if False:
        i = 10
        return i + 15
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.DeleteCdnKeyRequest(name='name_value')
    operation = client.delete_cdn_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)