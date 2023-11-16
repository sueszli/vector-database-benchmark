from google.cloud import deploy_v1

def sample_ignore_job():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.IgnoreJobRequest(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
    response = client.ignore_job(request=request)
    print(response)