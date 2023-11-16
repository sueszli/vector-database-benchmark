from google.cloud import domains_v1beta1

def sample_retrieve_transfer_parameters():
    if False:
        return 10
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.RetrieveTransferParametersRequest(domain_name='domain_name_value', location='location_value')
    response = client.retrieve_transfer_parameters(request=request)
    print(response)