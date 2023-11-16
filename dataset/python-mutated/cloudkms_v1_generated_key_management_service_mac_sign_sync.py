from google.cloud import kms_v1

def sample_mac_sign():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.MacSignRequest(name='name_value', data=b'data_blob')
    response = client.mac_sign(request=request)
    print(response)