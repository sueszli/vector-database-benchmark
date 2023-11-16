from google.cloud import tasks_v2beta2

def sample_list_tasks():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)