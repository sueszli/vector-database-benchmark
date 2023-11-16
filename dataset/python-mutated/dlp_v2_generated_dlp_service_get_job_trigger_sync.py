from google.cloud import dlp_v2

def sample_get_job_trigger():
    if False:
        print('Hello World!')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.GetJobTriggerRequest(name='name_value')
    response = client.get_job_trigger(request=request)
    print(response)