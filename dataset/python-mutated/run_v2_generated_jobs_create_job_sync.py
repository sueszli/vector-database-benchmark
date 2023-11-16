from google.cloud import run_v2

def sample_create_job():
    if False:
        return 10
    client = run_v2.JobsClient()
    job = run_v2.Job()
    job.template.template.max_retries = 1187
    request = run_v2.CreateJobRequest(parent='parent_value', job=job, job_id='job_id_value')
    operation = client.create_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)