from google.cloud import retail_v2beta

def sample_list_controls():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ControlServiceClient()
    request = retail_v2beta.ListControlsRequest(parent='parent_value')
    page_result = client.list_controls(request=request)
    for response in page_result:
        print(response)