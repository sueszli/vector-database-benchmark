from google.cloud import tasks_v2beta2

def sample_create_task():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.CreateTaskRequest(parent='parent_value')
    response = client.create_task(request=request)
    print(response)