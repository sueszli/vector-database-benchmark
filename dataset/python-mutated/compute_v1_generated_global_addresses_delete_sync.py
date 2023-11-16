from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.GlobalAddressesClient()
    request = compute_v1.DeleteGlobalAddressRequest(address='address_value', project='project_value')
    response = client.delete(request=request)
    print(response)