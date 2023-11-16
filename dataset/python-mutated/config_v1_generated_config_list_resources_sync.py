from google.cloud import config_v1

def sample_list_resources():
    if False:
        while True:
            i = 10
    client = config_v1.ConfigClient()
    request = config_v1.ListResourcesRequest(parent='parent_value')
    page_result = client.list_resources(request=request)
    for response in page_result:
        print(response)