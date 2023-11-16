from google.cloud.video import stitcher_v1

def sample_get_vod_ad_tag_detail():
    if False:
        return 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetVodAdTagDetailRequest(name='name_value')
    response = client.get_vod_ad_tag_detail(request=request)
    print(response)