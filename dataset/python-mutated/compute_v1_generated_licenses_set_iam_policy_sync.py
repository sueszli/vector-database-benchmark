from google.cloud import compute_v1

def sample_set_iam_policy():
    if False:
        return 10
    client = compute_v1.LicensesClient()
    request = compute_v1.SetIamPolicyLicenseRequest(project='project_value', resource='resource_value')
    response = client.set_iam_policy(request=request)
    print(response)