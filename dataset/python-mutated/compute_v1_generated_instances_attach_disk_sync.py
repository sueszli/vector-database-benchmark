from google.cloud import compute_v1

def sample_attach_disk():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.AttachDiskInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.attach_disk(request=request)
    print(response)