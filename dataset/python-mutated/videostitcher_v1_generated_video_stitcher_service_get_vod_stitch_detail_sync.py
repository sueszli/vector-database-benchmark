from google.cloud.video import stitcher_v1

def sample_get_vod_stitch_detail():
    if False:
        for i in range(10):
            print('nop')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.GetVodStitchDetailRequest(name='name_value')
    response = client.get_vod_stitch_detail(request=request)
    print(response)