from google.cloud import kms_v1

def sample_generate_random_bytes():
    if False:
        i = 10
        return i + 15
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.GenerateRandomBytesRequest()
    response = client.generate_random_bytes(request=request)
    print(response)