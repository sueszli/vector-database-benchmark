from google.cloud import vpcaccess_v1

def sample_delete_connector():
    if False:
        return 10
    client = vpcaccess_v1.VpcAccessServiceClient()
    request = vpcaccess_v1.DeleteConnectorRequest(name='name_value')
    operation = client.delete_connector(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)