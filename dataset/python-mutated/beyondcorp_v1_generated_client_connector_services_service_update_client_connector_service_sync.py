from google.cloud import beyondcorp_clientconnectorservices_v1

def sample_update_client_connector_service():
    if False:
        while True:
            i = 10
    client = beyondcorp_clientconnectorservices_v1.ClientConnectorServicesServiceClient()
    client_connector_service = beyondcorp_clientconnectorservices_v1.ClientConnectorService()
    client_connector_service.name = 'name_value'
    client_connector_service.ingress.config.transport_protocol = 'TCP'
    client_connector_service.ingress.config.destination_routes.address = 'address_value'
    client_connector_service.ingress.config.destination_routes.netmask = 'netmask_value'
    client_connector_service.egress.peered_vpc.network_vpc = 'network_vpc_value'
    request = beyondcorp_clientconnectorservices_v1.UpdateClientConnectorServiceRequest(client_connector_service=client_connector_service)
    operation = client.update_client_connector_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)