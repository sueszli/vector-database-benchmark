from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        return 10
    client = compute_v1.VpnTunnelsClient()
    request = compute_v1.AggregatedListVpnTunnelsRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)