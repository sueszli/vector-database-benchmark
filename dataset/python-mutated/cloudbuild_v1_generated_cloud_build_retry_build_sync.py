from google.cloud.devtools import cloudbuild_v1

def sample_retry_build():
    if False:
        while True:
            i = 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.RetryBuildRequest(project_id='project_id_value', id='id_value')
    operation = client.retry_build(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)