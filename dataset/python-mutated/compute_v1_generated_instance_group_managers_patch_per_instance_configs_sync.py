from google.cloud import compute_v1

def sample_patch_per_instance_configs():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.PatchPerInstanceConfigsInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.patch_per_instance_configs(request=request)
    print(response)