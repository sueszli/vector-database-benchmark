from google.cloud import kms_v1

def sample_create_key_ring():
    if False:
        while True:
            i = 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.CreateKeyRingRequest(parent='parent_value', key_ring_id='key_ring_id_value')
    response = client.create_key_ring(request=request)
    print(response)