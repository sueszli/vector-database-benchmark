from google.cloud import deploy_v1

def sample_get_job_run():
    if False:
        print('Hello World!')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetJobRunRequest(name='name_value')
    response = client.get_job_run(request=request)
    print(response)