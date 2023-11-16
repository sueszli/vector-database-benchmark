from google.cloud import compute_v1

def sample_set_service_account():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetServiceAccountInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_service_account(request=request)
    print(response)