from google.cloud import compute_v1

def sample_resize():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.ResizeInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', size=443, zone='zone_value')
    response = client.resize(request=request)
    print(response)