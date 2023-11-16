from google.cloud.video import stitcher_v1

def sample_get_cdn_key():
    if False:
        print('Hello World!')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetCdnKeyRequest(name='name_value')
    response = client.get_cdn_key(request=request)
    print(response)