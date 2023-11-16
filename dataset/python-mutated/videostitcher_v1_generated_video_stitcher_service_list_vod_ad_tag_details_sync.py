from google.cloud.video import stitcher_v1

def sample_list_vod_ad_tag_details():
    if False:
        while True:
            i = 10
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.ListVodAdTagDetailsRequest(parent='parent_value')
    page_result = client.list_vod_ad_tag_details(request=request)
    for response in page_result:
        print(response)