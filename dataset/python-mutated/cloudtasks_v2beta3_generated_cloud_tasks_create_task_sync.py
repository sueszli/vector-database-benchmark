from google.cloud import tasks_v2beta3

def sample_create_task():
    if False:
        print('Hello World!')
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.CreateTaskRequest(parent='parent_value')
    response = client.create_task(request=request)
    print(response)