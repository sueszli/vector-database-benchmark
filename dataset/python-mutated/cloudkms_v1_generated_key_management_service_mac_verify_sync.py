from google.cloud import kms_v1

def sample_mac_verify():
    if False:
        print('Hello World!')
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.MacVerifyRequest(name='name_value', data=b'data_blob', mac=b'mac_blob')
    response = client.mac_verify(request=request)
    print(response)