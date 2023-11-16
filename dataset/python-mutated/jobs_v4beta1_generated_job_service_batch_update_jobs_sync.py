from google.cloud import talent_v4beta1

def sample_batch_update_jobs():
    if False:
        print('Hello World!')
    client = talent_v4beta1.JobServiceClient()
    jobs = talent_v4beta1.Job()
    jobs.company = 'company_value'
    jobs.requisition_id = 'requisition_id_value'
    jobs.title = 'title_value'
    jobs.description = 'description_value'
    request = talent_v4beta1.BatchUpdateJobsRequest(parent='parent_value', jobs=jobs)
    operation = client.batch_update_jobs(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)