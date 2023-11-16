from google.cloud import deploy_v1

def sample_retry_job():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.RetryJobRequest(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
    response = client.retry_job(request=request)
    print(response)