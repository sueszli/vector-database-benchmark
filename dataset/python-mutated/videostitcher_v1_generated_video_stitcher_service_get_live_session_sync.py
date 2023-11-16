from google.cloud.video import stitcher_v1

def sample_get_live_session():
    if False:
        i = 10
        return i + 15
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetLiveSessionRequest(name='name_value')
    response = client.get_live_session(request=request)
    print(response)