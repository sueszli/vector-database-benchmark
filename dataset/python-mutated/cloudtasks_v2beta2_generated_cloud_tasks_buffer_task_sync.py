from google.cloud import tasks_v2beta2

def sample_buffer_task():
    if False:
        print('Hello World!')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.BufferTaskRequest(queue='queue_value')
    response = client.buffer_task(request=request)
    print(response)