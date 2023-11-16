from google.cloud import scheduler_v1

def sample_run_job():
    if False:
        while True:
            i = 10
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.RunJobRequest(name='name_value')
    response = client.run_job(request=request)
    print(response)