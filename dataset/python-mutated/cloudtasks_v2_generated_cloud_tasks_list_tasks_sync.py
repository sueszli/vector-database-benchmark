from google.cloud import tasks_v2

def sample_list_tasks():
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.ListTasksRequest(parent='parent_value')
    page_result = client.list_tasks(request=request)
    for response in page_result:
        print(response)