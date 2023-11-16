from google.cloud.devtools import cloudbuild_v1

def sample_update_build_trigger():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    trigger = cloudbuild_v1.BuildTrigger()
    trigger.autodetect = True
    request = cloudbuild_v1.UpdateBuildTriggerRequest(project_id='project_id_value', trigger_id='trigger_id_value', trigger=trigger)
    response = client.update_build_trigger(request=request)
    print(response)