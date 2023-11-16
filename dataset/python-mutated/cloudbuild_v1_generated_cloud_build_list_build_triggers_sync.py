from google.cloud.devtools import cloudbuild_v1

def sample_list_build_triggers():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ListBuildTriggersRequest(project_id='project_id_value')
    page_result = client.list_build_triggers(request=request)
    for response in page_result:
        print(response)