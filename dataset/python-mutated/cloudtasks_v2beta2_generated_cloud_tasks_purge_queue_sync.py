from google.cloud import tasks_v2beta2

def sample_purge_queue():
    if False:
        return 10
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.PurgeQueueRequest(name='name_value')
    response = client.purge_queue(request=request)
    print(response)