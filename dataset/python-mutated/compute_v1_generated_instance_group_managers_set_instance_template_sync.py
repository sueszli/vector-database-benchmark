from google.cloud import compute_v1

def sample_set_instance_template():
    if False:
        return 10
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.SetInstanceTemplateInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    response = client.set_instance_template(request=request)
    print(response)