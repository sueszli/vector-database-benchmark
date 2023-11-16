from google.cloud import tasks_v2beta3

def sample_delete_task():
    if False:
        for i in range(10):
            print('nop')
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.DeleteTaskRequest(name='name_value')
    client.delete_task(request=request)