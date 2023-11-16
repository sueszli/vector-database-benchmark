from google.cloud import dlp_v2

def sample_create_job_trigger():
    if False:
        return 10
    client = dlp_v2.DlpServiceClient()
    job_trigger = dlp_v2.JobTrigger()
    job_trigger.status = 'CANCELLED'
    request = dlp_v2.CreateJobTriggerRequest(parent='parent_value', job_trigger=job_trigger)
    response = client.create_job_trigger(request=request)
    print(response)