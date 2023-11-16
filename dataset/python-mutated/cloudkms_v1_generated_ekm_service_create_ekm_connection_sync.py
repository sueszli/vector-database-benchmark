from google.cloud import kms_v1

def sample_create_ekm_connection():
    if False:
        for i in range(10):
            print('nop')
    client = kms_v1.EkmServiceClient()
    request = kms_v1.CreateEkmConnectionRequest(parent='parent_value', ekm_connection_id='ekm_connection_id_value')
    response = client.create_ekm_connection(request=request)
    print(response)