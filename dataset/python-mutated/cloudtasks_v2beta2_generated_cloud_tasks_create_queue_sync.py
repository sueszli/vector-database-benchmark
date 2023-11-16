from google.cloud import tasks_v2beta2

def sample_create_queue():
    if False:
        print('Hello World!')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.CreateQueueRequest(parent='parent_value')
    response = client.create_queue(request=request)
    print(response)