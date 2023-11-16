from google.cloud import run_v2

def sample_get_job():
    if False:
        i = 10
        return i + 15
    client = run_v2.JobsClient()
    request = run_v2.GetJobRequest(name='name_value')
    response = client.get_job(request=request)
    print(response)