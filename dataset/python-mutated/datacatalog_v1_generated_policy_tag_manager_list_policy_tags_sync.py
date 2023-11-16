from google.cloud import datacatalog_v1

def sample_list_policy_tags():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.ListPolicyTagsRequest(parent='parent_value')
    page_result = client.list_policy_tags(request=request)
    for response in page_result:
        print(response)