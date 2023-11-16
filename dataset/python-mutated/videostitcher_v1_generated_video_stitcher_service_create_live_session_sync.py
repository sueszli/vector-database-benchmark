from google.cloud.video import stitcher_v1

def sample_create_live_session():
    if False:
        print('Hello World!')
    client = stitcher_v1.VideoStitcherServiceClient()
    live_session = stitcher_v1.LiveSession()
    live_session.live_config = 'live_config_value'
    request = stitcher_v1.CreateLiveSessionRequest(parent='parent_value', live_session=live_session)
    response = client.create_live_session(request=request)
    print(response)