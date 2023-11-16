from google.cloud.security import privateca_v1

def sample_delete_ca_pool():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.DeleteCaPoolRequest(name='name_value')
    operation = client.delete_ca_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)