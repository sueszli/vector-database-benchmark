from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.TargetInstancesClient()
    request = compute_v1.GetTargetInstanceRequest(project='project_value', target_instance='target_instance_value', zone='zone_value')
    response = client.get(request=request)
    print(response)