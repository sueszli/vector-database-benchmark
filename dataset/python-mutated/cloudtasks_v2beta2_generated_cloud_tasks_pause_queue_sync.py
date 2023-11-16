from google.cloud import tasks_v2beta2

def sample_pause_queue():
    if False:
        for i in range(10):
            print('nop')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.PauseQueueRequest(name='name_value')
    response = client.pause_queue(request=request)
    print(response)