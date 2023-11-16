from google.cloud.devtools import cloudbuild_v1

def sample_create_build_trigger():
    if False:
        i = 10
        return i + 15
    client = cloudbuild_v1.CloudBuildClient()
    trigger = cloudbuild_v1.BuildTrigger()
    trigger.autodetect = True
    request = cloudbuild_v1.CreateBuildTriggerRequest(project_id='project_id_value', trigger=trigger)
    response = client.create_build_trigger(request=request)
    print(response)