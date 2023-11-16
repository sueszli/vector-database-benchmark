from google.cloud import kms_v1

def sample_verify_connectivity():
    if False:
        while True:
            i = 10
    client = kms_v1.EkmServiceClient()
    request = kms_v1.VerifyConnectivityRequest(name='name_value')
    response = client.verify_connectivity(request=request)
    print(response)