from google.cloud import dlp_v2

def sample_update_job_trigger():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.UpdateJobTriggerRequest(name='name_value')
    response = client.update_job_trigger(request=request)
    print(response)