from google.cloud import tasks_v2beta3

def sample_update_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.UpdateQueueRequest()
    response = client.update_queue(request=request)
    print(response)