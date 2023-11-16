from google.cloud import run_v2

def sample_update_job():
    if False:
        print('Hello World!')
    client = run_v2.JobsClient()
    job = run_v2.Job()
    job.template.template.max_retries = 1187
    request = run_v2.UpdateJobRequest(job=job)
    operation = client.update_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)