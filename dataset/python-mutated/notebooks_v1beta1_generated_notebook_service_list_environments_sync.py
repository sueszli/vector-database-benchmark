from google.cloud import notebooks_v1beta1

def sample_list_environments():
    if False:
        print('Hello World!')
    client = notebooks_v1beta1.NotebookServiceClient()
    request = notebooks_v1beta1.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)