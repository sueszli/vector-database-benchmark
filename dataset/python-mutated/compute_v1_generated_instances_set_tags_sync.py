from google.cloud import compute_v1

def sample_set_tags():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SetTagsInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.set_tags(request=request)
    print(response)