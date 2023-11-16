from google.cloud import notebooks_v1

def sample_list_executions():
    if False:
        return 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.ListExecutionsRequest(parent='parent_value')
    page_result = client.list_executions(request=request)
    for response in page_result:
        print(response)