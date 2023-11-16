from google.cloud import channel_v1

def sample_update_customer():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    customer = channel_v1.Customer()
    customer.org_display_name = 'org_display_name_value'
    customer.domain = 'domain_value'
    request = channel_v1.UpdateCustomerRequest(customer=customer)
    response = client.update_customer(request=request)
    print(response)