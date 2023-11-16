from google.cloud.video import stitcher_v1

def sample_get_live_config():
    if False:
        while True:
            i = 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetLiveConfigRequest(name='name_value')
    response = client.get_live_config(request=request)
    print(response)