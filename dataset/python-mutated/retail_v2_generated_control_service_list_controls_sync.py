from google.cloud import retail_v2

def sample_list_controls():
    if False:
        return 10
    client = retail_v2.ControlServiceClient()
    request = retail_v2.ListControlsRequest(parent='parent_value')
    page_result = client.list_controls(request=request)
    for response in page_result:
        print(response)