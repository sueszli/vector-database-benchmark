from google.cloud import tasks_v2beta3

def sample_delete_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.DeleteQueueRequest(name='name_value')
    client.delete_queue(request=request)