from google.cloud import compute_v1

def sample_update_shielded_instance_config():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.UpdateShieldedInstanceConfigInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.update_shielded_instance_config(request=request)
    print(response)