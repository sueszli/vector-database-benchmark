from google.cloud import compute_v1

def sample_list_errors():
    if False:
        while True:
            i = 10
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.ListErrorsInstanceGroupManagersRequest(instance_group_manager='instance_group_manager_value', project='project_value', zone='zone_value')
    page_result = client.list_errors(request=request)
    for response in page_result:
        print(response)