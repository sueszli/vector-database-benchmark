from google.cloud import channel_v1

def sample_create_customer():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    customer = channel_v1.Customer()
    customer.org_display_name = 'org_display_name_value'
    customer.domain = 'domain_value'
    request = channel_v1.CreateCustomerRequest(parent='parent_value', customer=customer)
    response = client.create_customer(request=request)
    print(response)