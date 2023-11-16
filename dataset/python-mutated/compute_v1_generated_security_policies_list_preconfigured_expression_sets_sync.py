from google.cloud import compute_v1

def sample_list_preconfigured_expression_sets():
    if False:
        while True:
            i = 10
    client = compute_v1.SecurityPoliciesClient()
    request = compute_v1.ListPreconfiguredExpressionSetsSecurityPoliciesRequest(project='project_value')
    response = client.list_preconfigured_expression_sets(request=request)
    print(response)