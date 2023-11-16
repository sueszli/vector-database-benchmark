from google.cloud import scheduler_v1

def sample_delete_job():
    if False:
        for i in range(10):
            print('nop')
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.DeleteJobRequest(name='name_value')
    client.delete_job(request=request)