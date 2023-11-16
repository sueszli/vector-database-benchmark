from google.cloud.devtools import cloudbuild_v1

def sample_cancel_build():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.CancelBuildRequest(project_id='project_id_value', id='id_value')
    response = client.cancel_build(request=request)
    print(response)