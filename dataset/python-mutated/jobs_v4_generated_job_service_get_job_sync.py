from google.cloud import talent_v4

def sample_get_job():
    if False:
        for i in range(10):
            print('nop')
    client = talent_v4.JobServiceClient()
    request = talent_v4.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)