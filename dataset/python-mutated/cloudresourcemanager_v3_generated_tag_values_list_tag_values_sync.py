from google.cloud import resourcemanager_v3

def sample_list_tag_values():
    if False:
        print('Hello World!')
    client = resourcemanager_v3.TagValuesClient()
    request = resourcemanager_v3.ListTagValuesRequest(parent='parent_value')
    page_result = client.list_tag_values(request=request)
    for response in page_result:
        print(response)