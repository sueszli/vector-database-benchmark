from google.cloud.video import stitcher_v1

def sample_get_vod_session():
    if False:
        return 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetVodSessionRequest(name='name_value')
    response = client.get_vod_session(request=request)
    print(response)