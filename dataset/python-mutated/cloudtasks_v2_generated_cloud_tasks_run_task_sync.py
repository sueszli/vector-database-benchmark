from google.cloud import tasks_v2

def sample_run_task():
    if False:
        print('Hello World!')
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.RunTaskRequest(name='name_value')
    response = client.run_task(request=request)
    print(response)