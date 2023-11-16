from google.cloud import vpcaccess_v1

def sample_get_connector():
    if False:
        return 10
    client = vpcaccess_v1.VpcAccessServiceClient()
    request = vpcaccess_v1.GetConnectorRequest(name='name_value')
    response = client.get_connector(request=request)
    print(response)