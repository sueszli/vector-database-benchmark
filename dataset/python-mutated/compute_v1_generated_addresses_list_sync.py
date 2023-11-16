from google.cloud import compute_v1

def sample_list():
    if False:
        print('Hello World!')
    client = compute_v1.AddressesClient()
    request = compute_v1.ListAddressesRequest(project='project_value', region='region_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)