from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.TargetPoolsClient()
    request = compute_v1.AggregatedListTargetPoolsRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)