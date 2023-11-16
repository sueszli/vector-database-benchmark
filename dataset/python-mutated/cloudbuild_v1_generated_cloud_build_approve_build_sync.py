from google.cloud.devtools import cloudbuild_v1

def sample_approve_build():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ApproveBuildRequest(name='name_value')
    operation = client.approve_build(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)