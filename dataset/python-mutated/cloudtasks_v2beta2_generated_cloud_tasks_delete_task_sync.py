from google.cloud import tasks_v2beta2

def sample_delete_task():
    if False:
        return 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.DeleteTaskRequest(name='name_value')
    client.delete_task(request=request)