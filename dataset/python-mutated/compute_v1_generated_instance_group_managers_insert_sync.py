from google.cloud import compute_v1

def sample_insert():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.InsertInstanceGroupManagerRequest(project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)