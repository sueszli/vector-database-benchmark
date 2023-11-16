from google.cloud import scheduler_v1beta1

def sample_run_job():
    if False:
        i = 10
        return i + 15
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.RunJobRequest(name='name_value')
    response = client.run_job(request=request)
    print(response)