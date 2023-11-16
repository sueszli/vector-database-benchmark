from google.cloud import compute_v1

def sample_insert():
    if False:
        return 10
    client = compute_v1.TargetInstancesClient()
    request = compute_v1.InsertTargetInstanceRequest(project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)