from google.cloud import notebooks_v1

def sample_list_runtimes():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.ManagedNotebookServiceClient()
    request = notebooks_v1.ListRuntimesRequest(parent='parent_value')
    page_result = client.list_runtimes(request=request)
    for response in page_result:
        print(response)