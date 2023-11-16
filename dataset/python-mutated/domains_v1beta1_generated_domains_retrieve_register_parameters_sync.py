from google.cloud import domains_v1beta1

def sample_retrieve_register_parameters():
    if False:
        print('Hello World!')
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.RetrieveRegisterParametersRequest(domain_name='domain_name_value', location='location_value')
    response = client.retrieve_register_parameters(request=request)
    print(response)