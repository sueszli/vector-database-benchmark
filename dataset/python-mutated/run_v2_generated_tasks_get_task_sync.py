from google.cloud import run_v2

def sample_get_task():
    if False:
        i = 10
        return i + 15
    client = run_v2.TasksClient()
    request = run_v2.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)