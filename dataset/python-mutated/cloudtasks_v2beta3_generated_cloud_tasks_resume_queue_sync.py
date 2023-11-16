from google.cloud import tasks_v2beta3

def sample_resume_queue():
    if False:
        return 10
    client = tasks_v2beta3.CloudTasksClient()
    request = tasks_v2beta3.ResumeQueueRequest(name='name_value')
    response = client.resume_queue(request=request)
    print(response)