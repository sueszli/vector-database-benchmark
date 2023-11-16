from google.cloud import compute_v1

def sample_suspend():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.SuspendInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.suspend(request=request)
    print(response)