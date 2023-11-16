from google.cloud import notebooks_v1beta1

def sample_list_instances():
    if False:
        print('Hello World!')
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)