from google.cloud import compute_v1

def sample_list_errors():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionInstanceGroupManagersClient()
    request = compute_v1.ListErrorsRegionInstanceGroupManagersRequest(instance_group_manager='instance_group_manager_value', project='project_value', region='region_value')
    page_result = client.list_errors(request=request)
    for response in page_result:
        print(response)