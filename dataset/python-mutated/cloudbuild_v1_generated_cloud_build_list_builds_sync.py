from google.cloud.devtools import cloudbuild_v1

def sample_list_builds():
    if False:
        print('Hello World!')
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ListBuildsRequest(project_id='project_id_value')
    page_result = client.list_builds(request=request)
    for response in page_result:
        print(response)