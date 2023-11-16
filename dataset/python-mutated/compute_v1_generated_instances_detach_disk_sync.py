from google.cloud import compute_v1

def sample_detach_disk():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.DetachDiskInstanceRequest(device_name='device_name_value', instance='instance_value', project='project_value', zone='zone_value')
    response = client.detach_disk(request=request)
    print(response)