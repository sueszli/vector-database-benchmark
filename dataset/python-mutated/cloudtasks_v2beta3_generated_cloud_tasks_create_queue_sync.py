from google.cloud import tasks_v2beta3

def sample_create_queue():
    if False:
        return 10
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.CreateQueueRequest(parent='parent_value')
    response = client.create_queue(request=request)
    print(response)