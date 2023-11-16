from google.cloud import api_keys_v2

def sample_list_keys():
    if False:
        i = 10
        return i + 15
    client = api_keys_v2.ApiKeysClient()
    request = api_keys_v2.ListKeysRequest(parent='parent_value')
    page_result = client.list_keys(request=request)
    for response in page_result:
        print(response)