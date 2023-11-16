from google.cloud import compute_v1

def sample_start():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.StartInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.start(request=request)
    print(response)