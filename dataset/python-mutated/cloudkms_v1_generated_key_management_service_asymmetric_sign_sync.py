from google.cloud import kms_v1

def sample_asymmetric_sign():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.AsymmetricSignRequest(name='name_value')
    response = client.asymmetric_sign(request=request)
    print(response)