from google.cloud import scheduler_v1

def sample_update_job():
    if False:
        print('Hello World!')
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.UpdateJobRequest()
    response = client.update_job(request=request)
    print(response)