from google.cloud.video import stitcher_v1

def sample_list_cdn_keys():
    if False:
        for i in range(10):
            print('nop')
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.ListCdnKeysRequest(parent='parent_value')
    page_result = client.list_cdn_keys(request=request)
    for response in page_result:
        print(response)