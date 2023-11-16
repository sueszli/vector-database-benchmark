from google.cloud import scheduler_v1beta1

def sample_resume_job():
    if False:
        for i in range(10):
            print('nop')
    client = scheduler_v1beta1.CloudSchedulerClient()
    request = scheduler_v1beta1.ResumeJobRequest(name='name_value')
    response = client.resume_job(request=request)
    print(response)