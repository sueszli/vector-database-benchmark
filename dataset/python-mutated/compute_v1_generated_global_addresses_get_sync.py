from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.GlobalAddressesClient()
    request = compute_v1.GetGlobalAddressRequest(address='address_value', project='project_value')
    response = client.get(request=request)
    print(response)