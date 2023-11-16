from google.cloud import compute_v1

def sample_set_deletion_protection():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetDeletionProtectionInstanceRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.set_deletion_protection(request=request)
    print(response)