from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        return 10
    client = compute_v1.LicenseCodesClient()
    request = compute_v1.TestIamPermissionsLicenseCodeRequest(project='project_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)