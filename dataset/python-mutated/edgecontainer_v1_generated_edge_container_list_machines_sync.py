from google.cloud import edgecontainer_v1

def sample_list_machines():
    if False:
        print('Hello World!')
    client = edgecontainer_v1.EdgeContainerClient()
    request = edgecontainer_v1.ListMachinesRequest(parent='parent_value')
    page_result = client.list_machines(request=request)
    for response in page_result:
        print(response)