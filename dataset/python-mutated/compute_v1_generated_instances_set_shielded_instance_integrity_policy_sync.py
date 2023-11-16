from google.cloud import compute_v1

def sample_set_shielded_instance_integrity_policy():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstancesClient()
    request = compute_v1.SetShieldedInstanceIntegrityPolicyInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_shielded_instance_integrity_policy(request=request)
    print(response)