from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        print('Hello World!')
    client = compute_v1.InstanceGroupManagersClient()
    request = compute_v1.AggregatedListInstanceGroupManagersRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)