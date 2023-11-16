from google.cloud import vpcaccess_v1

def sample_create_connector():
    if False:
        for i in range(10):
            print('nop')
    client = vpcaccess_v1.VpcAccessServiceClient()
    request = vpcaccess_v1.CreateConnectorRequest(parent='parent_value', connector_id='connector_id_value')
    operation = client.create_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)