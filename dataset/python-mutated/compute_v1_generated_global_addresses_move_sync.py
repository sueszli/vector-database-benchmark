from google.cloud import compute_v1

def sample_move():
    if False:
        print('Hello World!')
    client = compute_v1.GlobalAddressesClient()
    request = compute_v1.MoveGlobalAddressRequest(address='address_value', project='project_value')
    response = client.move(request=request)
    print(response)