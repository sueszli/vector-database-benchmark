from google.cloud import dlp_v2

def sample_activate_job_trigger():
    if False:
        i = 10
        return i + 15
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ActivateJobTriggerRequest(name='name_value')
    response = client.activate_job_trigger(request=request)
    print(response)