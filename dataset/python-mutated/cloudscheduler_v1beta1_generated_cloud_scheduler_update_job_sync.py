from google.cloud import scheduler_v1beta1

def sample_update_job():
    if False:
        return 10
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.UpdateJobRequest()
    response = client.update_job(request=request)
    print(response)