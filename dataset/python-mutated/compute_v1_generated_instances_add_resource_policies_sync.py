from google.cloud import compute_v1

def sample_add_resource_policies():
    if False:
        return 10
    client = compute_v1.InstancesClient()
    request = compute_v1.AddResourcePoliciesInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.add_resource_policies(request=request)
    print(response)