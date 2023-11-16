from google.cloud import tasks_v2beta2

def sample_get_task():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)