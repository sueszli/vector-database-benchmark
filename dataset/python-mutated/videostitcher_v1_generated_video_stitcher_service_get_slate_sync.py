from google.cloud.video import stitcher_v1

def sample_get_slate():
    if False:
        print('Hello World!')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetSlateRequest(name='name_value')
    response = client.get_slate(request=request)
    print(response)