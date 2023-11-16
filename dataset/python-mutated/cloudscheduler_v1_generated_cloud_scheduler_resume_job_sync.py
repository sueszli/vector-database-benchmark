from google.cloud import scheduler_v1

def sample_resume_job():
    if False:
        while True:
            i = 10
    client = scheduler_v1.CloudSchedulerClient()
    request = scheduler_v1.ResumeJobRequest(name='name_value')
    response = client.resume_job(request=request)
    print(response)