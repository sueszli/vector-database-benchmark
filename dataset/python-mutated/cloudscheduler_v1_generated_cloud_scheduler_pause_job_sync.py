from google.cloud import scheduler_v1

def sample_pause_job():
    if False:
        return 10
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.PauseJobRequest(name='name_value')
    response = client.pause_job(request=request)
    print(response)