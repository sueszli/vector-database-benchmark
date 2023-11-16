from google.cloud import resourcemanager_v3

def sample_list_tag_holds():
    if False:
        while True:
            i = 10
    client = resourcemanager_v3.TagHoldsClient()
    request = resourcemanager_v3.ListTagHoldsRequest(parent='parent_value')
    page_result = client.list_tag_holds(request=request)
    for response in page_result:
        print(response)