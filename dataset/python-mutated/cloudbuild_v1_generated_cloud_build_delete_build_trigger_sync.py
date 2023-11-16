from google.cloud.devtools import cloudbuild_v1

def sample_delete_build_trigger():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.DeleteBuildTriggerRequest(project_id='project_id_value', trigger_id='trigger_id_value')
    client.delete_build_trigger(request=request)