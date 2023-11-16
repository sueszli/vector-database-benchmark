from google.cloud.devtools import cloudbuild_v1

def sample_get_build():
    if False:
        print('Hello World!')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.GetBuildRequest(project_id='project_id_value', id='id_value')
    response = client.get_build(request=request)
    print(response)