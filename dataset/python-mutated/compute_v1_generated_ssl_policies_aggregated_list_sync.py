from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SslPoliciesClient()
    request = compute_v1.AggregatedListSslPoliciesRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)