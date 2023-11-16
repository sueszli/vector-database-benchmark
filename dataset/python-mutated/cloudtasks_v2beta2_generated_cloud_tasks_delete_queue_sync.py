from google.cloud import tasks_v2beta2

def sample_delete_queue():
    if False:
        print('Hello World!')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.DeleteQueueRequest(name='name_value')
    client.delete_queue(request=request)