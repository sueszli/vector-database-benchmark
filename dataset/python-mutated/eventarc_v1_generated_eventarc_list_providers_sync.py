from google.cloud import eventarc_v1

def sample_list_providers():
    if False:
        for i in range(10):
            print('nop')
    client = eventarc_v1.EventarcClient()
    request = eventarc_v1.ListProvidersRequest(parent='parent_value')
    page_result = client.list_providers(request=request)
    for response in page_result:
        print(response)