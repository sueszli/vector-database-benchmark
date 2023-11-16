from google.cloud import domains_v1

def sample_retrieve_transfer_parameters():
    if False:
        i = 10
        return i + 15
    client = domains_v1.DomainsClient()
    request = domains_v1.RetrieveTransferParametersRequest(domain_name='domain_name_value', location='location_value')
    response = client.retrieve_transfer_parameters(request=request)
    print(response)