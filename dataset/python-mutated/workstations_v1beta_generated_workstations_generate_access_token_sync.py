from google.cloud import workstations_v1beta

def sample_generate_access_token():
    if False:
        print('Hello World!')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.GenerateAccessTokenRequest(workstation='workstation_value')
    response = client.generate_access_token(request=request)
    print(response)