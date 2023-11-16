from google.cloud.video import stitcher_v1

def sample_update_cdn_key():
    if False:
        for i in range(10):
            print('nop')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.UpdateCdnKeyRequest()
    operation = client.update_cdn_key(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)