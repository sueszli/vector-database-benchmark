from google.cloud.devtools import cloudbuild_v1

def sample_create_build():
    if False:
        while True:
            i = 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.CreateBuildRequest(project_id='project_id_value')
    operation = client.create_build(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)