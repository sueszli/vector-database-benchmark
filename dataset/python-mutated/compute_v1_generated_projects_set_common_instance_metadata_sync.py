from google.cloud import compute_v1

def sample_set_common_instance_metadata():
    if False:
        print('Hello World!')
    client = compute_v1.ProjectsClient()
    request = compute_v1.SetCommonInstanceMetadataProjectRequest(project='project_value')
    response = client.set_common_instance_metadata(request=request)
    print(response)