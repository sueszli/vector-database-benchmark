from google.cloud import notebooks_v2

def sample_list_instances():
    if False:
        i = 10
        return i + 15
    client = notebooks_v2.NotebookServiceClient()
    request = notebooks_v2.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)