from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        while True:
            i = 10
    client = compute_v1.ResourcePoliciesClient()
    request = compute_v1.TestIamPermissionsResourcePolicyRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)