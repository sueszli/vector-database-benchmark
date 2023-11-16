from google.cloud import tasks_v2

def sample_delete_queue():
    if False:
        return 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.DeleteQueueRequest(name='name_value')
    client.delete_queue(request=request)