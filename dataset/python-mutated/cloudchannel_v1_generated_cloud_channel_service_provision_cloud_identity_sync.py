from google.cloud import channel_v1

def sample_provision_cloud_identity():
    if False:
        i = 10
        return i + 15
    client = channel_v1.CloudChannelServiceClient()
    request = channel_v1.ProvisionCloudIdentityRequest(customer='customer_value')
    operation = client.provision_cloud_identity(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)