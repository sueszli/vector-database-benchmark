from google.cloud import beyondcorp_clientconnectorservices_v1

def sample_create_client_connector_service():
    if False:
        return 10
    client = beyondcorp_clientconnectorservices_v1.ClientConnectorServicesServiceClient()
    client_connector_service = beyondcorp_clientconnectorservices_v1.ClientConnectorService()
    client_connector_service.name = 'name_value'
    client_connector_service.ingress.config.transport_protocol = 'TCP'
    client_connector_service.ingress.config.destination_routes.address = 'address_value'
    client_connector_service.ingress.config.destination_routes.netmask = 'netmask_value'
    client_connector_service.egress.peered_vpc.network_vpc = 'network_vpc_value'
    request = beyondcorp_clientconnectorservices_v1.CreateClientConnectorServiceRequest(parent='parent_value', client_connector_service=client_connector_service)
    operation = client.create_client_connector_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)