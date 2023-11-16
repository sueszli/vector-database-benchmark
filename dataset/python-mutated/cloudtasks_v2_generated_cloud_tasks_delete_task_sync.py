from google.cloud import tasks_v2

def sample_delete_task():
    if False:
        print('Hello World!')
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.DeleteTaskRequest(name='name_value')
    client.delete_task(request=request)