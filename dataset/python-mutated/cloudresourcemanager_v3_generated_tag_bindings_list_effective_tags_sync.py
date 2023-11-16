from google.cloud import resourcemanager_v3

def sample_list_effective_tags():
    if False:
        for i in range(10):
            print('nop')
    client = resourcemanager_v3.TagBindingsClient()
    request = resourcemanager_v3.ListEffectiveTagsRequest(parent='parent_value')
    page_result = client.list_effective_tags(request=request)
    for response in page_result:
        print(response)