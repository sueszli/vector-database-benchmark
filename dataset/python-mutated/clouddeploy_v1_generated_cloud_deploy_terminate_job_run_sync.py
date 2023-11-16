from google.cloud import deploy_v1

def sample_terminate_job_run():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.TerminateJobRunRequest(name='name_value')
    response = client.terminate_job_run(request=request)
    print(response)