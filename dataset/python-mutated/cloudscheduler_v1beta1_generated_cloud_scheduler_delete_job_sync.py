from google.cloud import scheduler_v1beta1

def sample_delete_job():
    if False:
        print('Hello World!')
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.DeleteJobRequest(name='name_value')
    client.delete_job(request=request)