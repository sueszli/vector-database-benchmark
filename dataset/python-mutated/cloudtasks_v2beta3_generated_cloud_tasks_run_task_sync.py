from google.cloud import tasks_v2beta3

def sample_run_task():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.RunTaskRequest(name='name_value')
    response = client.run_task(request=request)
    print(response)