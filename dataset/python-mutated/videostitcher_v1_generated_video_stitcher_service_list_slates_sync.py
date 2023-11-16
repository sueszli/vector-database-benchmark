from google.cloud.video import stitcher_v1

def sample_list_slates():
    if False:
        i = 10
        return i + 15
    client = stitcher_v1.VideoStitcherServiceClient()
    request = stitcher_v1.ListSlatesRequest(parent='parent_value')
    page_result = client.list_slates(request=request)
    for response in page_result:
        print(response)