from google.cloud import compute_v1

def sample_list():
    if False:
        return 10
    client = compute_v1.TargetHttpProxiesClient()
    request = compute_v1.ListTargetHttpProxiesRequest(project='project_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)