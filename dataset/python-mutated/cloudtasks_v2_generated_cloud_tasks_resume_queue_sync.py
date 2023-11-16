from google.cloud import tasks_v2

def sample_resume_queue():
    if False:
        i = 10
        return i + 15
    client = tasks_v2.CloudTasksClient()
    request = tasks_v2.ResumeQueueRequest(name='name_value')
    response = client.resume_queue(request=request)
    print(response)