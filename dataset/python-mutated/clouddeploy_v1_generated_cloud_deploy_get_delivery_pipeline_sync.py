from google.cloud import deploy_v1

def sample_get_delivery_pipeline():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetDeliveryPipelineRequest(name='name_value')
    response = client.get_delivery_pipeline(request=request)
    print(response)