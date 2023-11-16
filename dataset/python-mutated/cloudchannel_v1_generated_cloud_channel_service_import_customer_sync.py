from google.cloud import channel_v1

def sample_import_customer():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ImportCustomerRequest(domain='domain_value', parent='parent_value', overwrite_if_exists=True)
    response = client.import_customer(request=request)
    print(response)