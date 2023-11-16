from google.cloud import dataplex_v1

def sample_list_content():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.ContentServiceClient()
    request = dataplex_v1.ListContentRequest(parent='parent_value')
    page_result = client.list_content(request=request)
    for response in page_result:
        print(response)