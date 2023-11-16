from google.cloud import dataplex_v1

def sample_list_tasks():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)