from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        return 10
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.AggregatedListVpnGatewaysRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)