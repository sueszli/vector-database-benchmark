from google.cloud import talent_v4

def sample_update_job():
    if False:
        print('Hello World!')
    client = talent_v4.JobServiceClient()
    job = talent_v4.Job()
    job.company = 'company_value'
    job.requisition_id = 'requisition_id_value'
    job.title = 'title_value'
    job.description = 'description_value'
    request = talent_v4.UpdateJobRequest(job=job)
    response = client.update_job(request=request)
    print(response)