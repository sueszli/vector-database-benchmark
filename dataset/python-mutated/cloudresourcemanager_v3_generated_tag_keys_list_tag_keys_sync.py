from google.cloud import resourcemanager_v3

def sample_list_tag_keys():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagKeysClient()
    request = resourcemanager_v3.ListTagKeysRequest(parent='parent_value')
    page_result = client.list_tag_keys(request=request)
    for response in page_result:
        print(response)