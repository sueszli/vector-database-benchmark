from google.cloud import compute_v1

def sample_set_metadata():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstancesClient()
    request = compute_v1.SetMetadataInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_metadata(request=request)
    print(response)