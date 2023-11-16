from google.cloud import tasks_v2

def sample_update_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.UpdateQueueRequest()
    response = client.update_queue(request=request)
    print(response)