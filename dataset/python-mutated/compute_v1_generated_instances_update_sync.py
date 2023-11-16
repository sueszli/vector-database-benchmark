from google.cloud import compute_v1

def sample_update():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstancesClient()
    request = compute_v1.UpdateInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.update(request=request)
    print(response)