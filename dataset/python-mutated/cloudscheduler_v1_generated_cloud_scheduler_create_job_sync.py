from google.cloud import scheduler_v1

def sample_create_job():
    if False:
        i = 10
        return i + 15
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.CreateJobRequest(parent='parent_value')
    response = client.create_job(request=request)
    print(response)