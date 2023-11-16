from google.cloud import compute_v1

def sample_move():
    if False:
        print('Hello World!')
    client = compute_v1.AddressesClient()
    request = compute_v1.MoveAddressRequest(address='address_value', project='project_value', region='region_value')
    response = client.move(request=request)
    print(response)