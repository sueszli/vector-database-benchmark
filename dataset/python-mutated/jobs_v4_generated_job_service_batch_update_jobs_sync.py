from google.cloud import talent_v4

def sample_batch_update_jobs():
    if False:
        i = 10
        return i + 15
    client = talent_v4.JobServiceClient()
    jobs = talent_v4.Job()
    jobs.company = 'company_value'
    jobs.requisition_id = 'requisition_id_value'
    jobs.title = 'title_value'
    jobs.description = 'description_value'
    request = talent_v4.BatchUpdateJobsRequest(parent='parent_value', jobs=jobs)
    operation = client.batch_update_jobs(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)