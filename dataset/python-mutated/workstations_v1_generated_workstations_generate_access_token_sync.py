from google.cloud import workstations_v1

def sample_generate_access_token():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.GenerateAccessTokenRequest(workstation='workstation_value')
    response = client.generate_access_token(request=request)
    print(response)