from google.cloud import tasks_v2beta3

def sample_get_queue():
    if False:
        print('Hello World!')
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.GetQueueRequest(name='name_value')
    response = client.get_queue(request=request)
    print(response)