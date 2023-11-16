from google.cloud import tasks_v2beta3

def sample_pause_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.PauseQueueRequest(name='name_value')
    response = client.pause_queue(request=request)
    print(response)