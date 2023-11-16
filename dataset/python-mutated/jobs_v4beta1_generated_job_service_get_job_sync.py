from google.cloud import talent_v4beta1

def sample_get_job():
    if False:
        return 10
    client = talent_v4beta1.JobServiceClient()
    request = talent_v4beta1.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)