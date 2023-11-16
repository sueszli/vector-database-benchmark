from google.cloud import batch_v1alpha

def sample_list_tasks():
    if False:
        print('Hello World!')
    client = batch_v1alpha.BatchServiceClient()
    request = batch_v1alpha.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)