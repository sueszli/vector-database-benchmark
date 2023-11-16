from google.cloud import compute_v1

def sample_start_with_encryption_key():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstancesClient()
    request = compute_v1.StartWithEncryptionKeyInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.start_with_encryption_key(request=request)
    print(response)