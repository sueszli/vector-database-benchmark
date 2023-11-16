from google.cloud import notebooks_v1

def sample_list_instances():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)