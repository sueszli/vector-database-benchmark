from google.cloud.video import stitcher_v1

def sample_get_live_ad_tag_detail():
    if False:
        return 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetLiveAdTagDetailRequest(name='name_value')
    response = client.get_live_ad_tag_detail(request=request)
    print(response)