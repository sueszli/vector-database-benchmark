from google.cloud import compute_v1

def sample_list():
    if False:
        print('Hello World!')
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.ListSecurityPoliciesRequest(project='project_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)