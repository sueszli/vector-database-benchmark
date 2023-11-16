from google.cloud import tasks_v2

def sample_pause_queue():
    if False:
        while True:
            i = 10
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.PauseQueueRequest(name='name_value')
    response = client.pause_queue(request=request)
    print(response)