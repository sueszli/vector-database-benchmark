from google.cloud import channel_v1

def sample_update_customer_repricing_config():
    if False:
        while True:
            i = 10
    client = channel_v1.CloudChannelServiceClient()
    customer_repricing_config = channel_v1.CustomerRepricingConfig()
    customer_repricing_config.repricing_config.rebilling_basis = 'DIRECT_CUSTOMER_COST'
    request = channel_v1.UpdateCustomerRepricingConfigRequest(customer_repricing_config=customer_repricing_config)
    response = client.update_customer_repricing_config(request=request)
    print(response)