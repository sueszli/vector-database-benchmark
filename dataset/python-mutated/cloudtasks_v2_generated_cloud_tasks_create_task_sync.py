from google.cloud import tasks_v2

def sample_create_task():
    if False:
        return 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.CreateTaskRequest(parent='parent_value')
    response = client.create_task(request=request)
    print(response)