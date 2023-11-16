from google.cloud.security import privateca_v1

def sample_create_ca_pool():
    if False:
        i = 10
        return i + 15
    client = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool = privateca_v1.CaPool()
    ca_pool.tier = 'DEVOPS'
    request = privateca_v1.CreateCaPoolRequest(parent='parent_value', ca_pool_id='ca_pool_id_value', ca_pool=ca_pool)
    operation = client.create_ca_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)