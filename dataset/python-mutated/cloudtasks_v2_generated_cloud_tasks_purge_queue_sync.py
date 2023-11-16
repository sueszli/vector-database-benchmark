from google.cloud import tasks_v2

def sample_purge_queue():
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.PurgeQueueRequest(name='name_value')
    response = client.purge_queue(request=request)
    print(response)