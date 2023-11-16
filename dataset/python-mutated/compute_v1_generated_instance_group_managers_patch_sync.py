from google.cloud import compute_v1

def sample_patch():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.PatchInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.patch(request=request)
    print(response)