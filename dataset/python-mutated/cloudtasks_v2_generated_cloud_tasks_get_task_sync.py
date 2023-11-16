from google.cloud import tasks_v2

def sample_get_task():
    if False:
        return 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)