from google.cloud import channel_v1

def sample_create_customer_repricing_config():
    if False:
        return 10
    client = channel_v1.CloudChannelServiceClient()
    customer_repricing_config = channel_v1.CustomerRepricingConfig()
    customer_repricing_config.repricing_config.rebilling_basis = 'DIRECT_CUSTOMER_COST'
    request = channel_v1.CreateCustomerRepricingConfigRequest(parent='parent_value', customer_repricing_config=customer_repricing_config)
    response = client.create_customer_repricing_config(request=request)
    print(response)