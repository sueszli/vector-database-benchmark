from google.cloud import compute_v1

def sample_reset():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.ResetInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.reset(request=request)
    print(response)