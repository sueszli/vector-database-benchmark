from google.cloud.security import privateca_v1

def sample_update_ca_pool():
    if False:
        while True:
            i = 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    ca_pool = privateca_v1.CaPool()
    ca_pool.tier = 'DEVOPS'
    request = privateca_v1.UpdateCaPoolRequest(ca_pool=ca_pool)
    operation = client.update_ca_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)