from google.cloud import compute_v1

def sample_set_instance_template():
    if False:
        return 10
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.SetInstanceTemplateRegionInstanceGroupManagerRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    response = client.set_instance_template(request=request)
    print(response)