from google.cloud import beyondcorp_appgateways_v1

def sample_create_app_gateway():
    if False:
        print('Hello World!')
    client = beyondcorp_appgateways_v1.AppGatewaysServiceClient()
    app_gateway = beyondcorp_appgateways_v1.AppGateway()
    app_gateway.name = 'name_value'
    app_gateway.type_ = 'TCP_PROXY'
    app_gateway.host_type = 'GCP_REGIONAL_MIG'
    request = beyondcorp_appgateways_v1.CreateAppGatewayRequest(parent='parent_value', app_gateway=app_gateway)
    operation = client.create_app_gateway(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)