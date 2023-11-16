from google.cloud import compute_v1

def sample_delete_access_config():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.DeleteAccessConfigInstanceRequest(access_config='access_config_value', instance='instance_value', network_interface='network_interface_value', project='project_value', zone='zone_value')
    response = client.delete_access_config(request=request)
    print(response)