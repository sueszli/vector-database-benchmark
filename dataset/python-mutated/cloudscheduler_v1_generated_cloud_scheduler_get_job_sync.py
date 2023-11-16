from google.cloud import scheduler_v1

def sample_get_job():
    if False:
        i = 10
        return i + 15
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)