from google.cloud import tasks_v2beta2

def sample_resume_queue():
    if False:
        print('Hello World!')
    client = tasks_v2beta2.CloudTasksClient()
    request = tasks_v2beta2.ResumeQueueRequest(name='name_value')
    response = client.resume_queue(request=request)
    print(response)