from google.cloud import scheduler_v1beta1

def sample_create_job():
    if False:
        for i in range(10):
            print('nop')
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.CreateJobRequest(parent='parent_value')
    response = client.create_job(request=request)
    print(response)