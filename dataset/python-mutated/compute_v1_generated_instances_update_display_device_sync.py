from google.cloud import compute_v1

def sample_update_display_device():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.UpdateDisplayDeviceInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.update_display_device(request=request)
    print(response)