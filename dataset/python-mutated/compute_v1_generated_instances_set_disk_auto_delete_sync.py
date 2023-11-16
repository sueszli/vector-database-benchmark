from google.cloud import compute_v1

def sample_set_disk_auto_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetDiskAutoDeleteInstanceRequest(auto_delete=True, device_name='device_name_value', instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_disk_auto_delete(request=request)
    print(response)