from google.cloud import tasks_v2beta3

def sample_get_task():
    if False:
        for i in range(10):
            print('nop')
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)