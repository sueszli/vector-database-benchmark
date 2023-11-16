from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.LicensesClient()
    request = compute_v1.DeleteLicenseRequest(license_='license__value', project='project_value')
    response = client.delete(request=request)
    print(response)