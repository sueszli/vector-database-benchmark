from google.cloud.video import stitcher_v1

def sample_create_vod_session():
    if False:
        i = 10
        return i + 15
    client = stitcher_v1.VideoStitcherServiceClient()
    vod_session = stitcher_v1.VodSession()
    vod_session.source_uri = 'source_uri_value'
    vod_session.ad_tag_uri = 'ad_tag_uri_value'
    vod_session.ad_tracking = 'SERVER'
    request = stitcher_v1.CreateVodSessionRequest(parent='parent_value', vod_session=vod_session)
    response = client.create_vod_session(request=request)
    print(response)