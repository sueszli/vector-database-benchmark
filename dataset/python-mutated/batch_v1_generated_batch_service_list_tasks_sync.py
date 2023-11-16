from google.cloud import batch_v1

def sample_list_tasks():
    if False:
        return 10
    client = batch_v1.BatchServiceClient()
    request = batch_v1.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)