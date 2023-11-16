from google.cloud import resourcemanager_v3

def sample_list_tag_bindings():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.TagBindingsClient()
    request = resourcemanager_v3.ListTagBindingsRequest(parent='parent_value')
    page_result = client.list_tag_bindings(request=request)
    for response in page_result:
        print(response)