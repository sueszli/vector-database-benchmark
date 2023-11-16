from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.VpnGatewaysClient()
    request = compute_v1.InsertVpnGatewayRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)