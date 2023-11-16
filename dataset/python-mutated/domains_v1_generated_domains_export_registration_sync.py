from google.cloud import domains_v1

def sample_export_registration():
    if False:
        while True:
            i = 10
    client = domains_v1.DomainsClient()
    request = domains_v1.ExportRegistrationRequest(name='name_value')
    operation = client.export_registration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)