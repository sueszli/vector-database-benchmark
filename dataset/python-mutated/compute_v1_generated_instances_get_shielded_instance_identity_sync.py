from google.cloud import compute_v1

def sample_get_shielded_instance_identity():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.GetShieldedInstanceIdentityInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.get_shielded_instance_identity(request=request)
    print(response)