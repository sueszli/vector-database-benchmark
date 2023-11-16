from google.cloud import tasks_v2

def sample_get_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.GetQueueRequest(name='name_value')
    response = client.get_queue(request=request)
    print(response)