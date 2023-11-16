from google.cloud import scheduler_v1beta1

def sample_pause_job():
    if False:
        for i in range(10):
            print('nop')
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.PauseJobRequest(name='name_value')
    response = client.pause_job(request=request)
    print(response)