from google.cloud import tasks_v2

def sample_create_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.CreateQueueRequest(parent='parent_value')
    response = client.create_queue(request=request)
    print(response)