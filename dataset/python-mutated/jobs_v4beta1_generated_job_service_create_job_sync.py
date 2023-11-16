from google.cloud import talent_v4beta1

def sample_create_job():
    if False:
        print('Hello World!')
    client = talent_v4beta1.JobServiceClient()
    job = talent_v4beta1.Job()
    job.company = 'company_value'
    job.requisition_id = 'requisition_id_value'
    job.title = 'title_value'
    job.description = 'description_value'
    request = talent_v4beta1.CreateJobRequest(parent='parent_value', job=job)
    response = client.create_job(request=request)
    print(response)