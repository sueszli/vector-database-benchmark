from google.cloud.devtools import cloudbuild_v1

def sample_get_build_trigger():
    if False:
        print('Hello World!')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.GetBuildTriggerRequest(project_id='project_id_value', trigger_id='trigger_id_value')
    response = client.get_build_trigger(request=request)
    print(response)