from google.cloud import talent_v4beta1

def sample_update_job():
    if False:
        while True:
            i = 10
    client = talent_v4beta1.JobServiceClient()
    job = talent_v4beta1.Job()
    job.company = 'company_value'
    job.requisition_id = 'requisition_id_value'
    job.title = 'title_value'
    job.description = 'description_value'
    request = talent_v4beta1.UpdateJobRequest(job=job)
    response = client.update_job(request=request)
    print(response)