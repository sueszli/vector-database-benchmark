from google.cloud import channel_v1

def sample_list_customer_repricing_configs():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ListCustomerRepricingConfigsRequest(parent='parent_value')
    page_result = client.list_customer_repricing_configs(request=request)
    for response in page_result:
        print(response)