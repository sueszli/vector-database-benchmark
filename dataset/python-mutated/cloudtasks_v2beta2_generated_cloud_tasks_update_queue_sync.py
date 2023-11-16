from google.cloud import tasks_v2beta2

def sample_update_queue():
    if False:
        print('Hello World!')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.UpdateQueueRequest()
    response = client.update_queue(request=request)
    print(response)