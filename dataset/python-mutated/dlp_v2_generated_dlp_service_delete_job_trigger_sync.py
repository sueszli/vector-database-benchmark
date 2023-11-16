from google.cloud import dlp_v2

def sample_delete_job_trigger():
    if False:
        while True:
            i = 10
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.DeleteJobTriggerRequest(name='name_value')
    client.delete_job_trigger(request=request)