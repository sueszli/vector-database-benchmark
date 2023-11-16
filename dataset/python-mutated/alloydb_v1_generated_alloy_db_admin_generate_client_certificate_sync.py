from google.cloud import alloydb_v1

def sample_generate_client_certificate():
    if False:
        while True:
            i = 10
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.GenerateClientCertificateRequest(parent='parent_value')
    response = client.generate_client_certificate(request=request)
    print(response)