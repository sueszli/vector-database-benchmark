from google.cloud import clouddms_v1

def sample_create_private_connection():
    if False:
        return 10
    client = clouddms_v1.DataMigrationServiceClient()
    private_connection = clouddms_v1.PrivateConnection()
    private_connection.vpc_peering_config.vpc_name = 'vpc_name_value'
    private_connection.vpc_peering_config.subnet = 'subnet_value'
    request = clouddms_v1.CreatePrivateConnectionRequest(parent='parent_value', private_connection_id='private_connection_id_value', private_connection=private_connection)
    operation = client.create_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)