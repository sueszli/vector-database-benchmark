from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.LicensesClient()
    request = compute_v1.GetLicenseRequest(license_='license__value', project='project_value')
    response = client.get(request=request)
    print(response)